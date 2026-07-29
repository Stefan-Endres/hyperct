"""Local mesh operations on 2D simplicial complexes.

All operations update the primal connectivity (``v.nn``) and the vertex
cache in-place.  They do **not** recompute dual cells; the caller (the
adaptive remesh driver or ``_retopologize``) is responsible for calling
``compute_vd`` after a batch of operations.

Triangles are inferred implicitly from the connectivity: a triangle is
any triple ``(v_a, v_b, v_c)`` such that every pair is mutually connected.
This matches how the ``Complex`` class stores 2-simplices via edges.

Edge data model
---------------
An edge is simply a pair of mutually-connected vertices.  Given an edge
``(v_i, v_j)``, the *opposite vertices* of its two adjacent triangles
are the vertices in ``v_i.nn & v_j.nn``.  In a manifold 2D mesh this set
has size 2 for interior edges and size 1 for boundary edges.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

from hyperct.remesh._quality import triangle_area, triangle_min_angle

# ---------------------------------------------------------------------------
# Vertex attribute carrying
# ---------------------------------------------------------------------------

# Attributes that should NOT be copied across when inserting a new vertex
# (they are recomputed by the dual/field pipeline).
_SKIP_ATTRS = frozenset({
    "x", "x_a", "nn", "hash", "index", "vd", "dual_vol",
    "dual_vol_phase", "is_interface", "interface_phases",
    "cache", "V",
    "f", "feasible",
    "check_min", "check_max",
    # Extensive quantities (total mass, per-phase masses) are handled
    # explicitly by each operation: split transfers a conservative
    # share FROM the endpoints (see edge_split_2d), collapse sums.
    # They must never be averaged like the intensive fields below.
    "m",
    "m_phase",
    # Phase is a categorical tag — handled explicitly at the end of
    # _carry_attrs (inherited from src_i verbatim).  Must not be
    # averaged by the numeric branch.
    "phase",
})


def _carry_attrs(dst, src_i, src_j=None) -> None:
    """Copy simulation attributes from ``src_i`` (and optionally ``src_j``)
    to ``dst``.

    For scalar and array attributes we average component-wise; for
    boolean attributes we take a logical AND; for everything else we
    copy from ``src_i``.  If only ``src_i`` is given, values are
    copied directly (used for collapse, where the surviving vertex
    is ``src_i``).

    Phase inheritance
    -----------------
    ``v.phase`` is a categorical tag — it can't be averaged.  We copy
    it unconditionally from ``src_i``, which has the following
    consequence on an *interface edge* split (``src_i.phase=0``,
    ``src_j.phase=1``):

        The new midpoint vertex inherits ``phase=0`` (the ``src_i``
        side), so the split edge becomes
            (v_i, v_m)        — same phase, interior edge
            (v_m, v_j)        — cross phase, NEW interface edge
            (v_m, v_k)        — k is an opposite vertex; its phase
                                determines whether this is an interface
                                or bulk edge

    This convention keeps the discrete interface a continuous chain
    while preserving the number of interface edges across the split
    (one cross-phase edge in, one cross-phase edge out, plus
    potentially one more from the new opposite-vertex connection).
    Callers that need a different convention should set ``phase``
    explicitly on the returned vertex.
    """
    # Snapshot source dicts so mutations of ``dst`` (which may alias
    # ``src_i`` in the collapse case ``_carry_attrs(v_i, v_i, v_j)``)
    # don't corrupt subsequent reads in this loop.
    src_i_snap = dict(getattr(src_i, "__dict__", {}))
    src_j_snap = (dict(getattr(src_j, "__dict__", {}))
                  if src_j is not None else None)

    # Capture the categorical phase tag BEFORE any mutation happens,
    # so the trailing override below still sees the correct source
    # value even when ``dst is src_i``.
    p_i_snap = src_i_snap.get("phase", None)

    for key, val_i in src_i_snap.items():
        if key in _SKIP_ATTRS or key.startswith("_"):
            continue
        if src_j_snap is not None:
            val_j = src_j_snap.get(key, val_i)
        else:
            val_j = val_i
        try:
            # Bool before int: isinstance(True, int) is True in Python,
            # so the bool check must come first.
            if isinstance(val_i, bool) and isinstance(val_j, bool):
                setattr(dst, key, bool(val_i and val_j))
            elif isinstance(val_i, np.ndarray) and isinstance(val_j, np.ndarray) \
                    and val_i.shape == val_j.shape:
                new_val = 0.5 * (val_i.astype(float) + val_j.astype(float))
                setattr(dst, key, new_val)
            elif isinstance(val_i, (int, float, np.integer, np.floating)) \
                    and isinstance(val_j, (int, float, np.integer, np.floating)):
                setattr(dst, key, 0.5 * (float(val_i) + float(val_j)))
            else:
                # Non-averageable types (strings, enums, etc.): copy from i.
                # copy.copy may fail on types with custom __copy__ — the
                # outer except drops the attribute in that case.
                setattr(dst, key, copy.copy(val_i))
        except (TypeError, ValueError, AttributeError):
            # Skip any attribute whose value can't be safely averaged
            # or copied.  This is deliberate: fields we genuinely care
            # about (``u``, ``p``, ``m``, ``phase``) go through the
            # typed branches above; the fallback only exists for
            # user-defined tags we don't recognise.
            pass

    # Restore the categorical phase tag from the captured snapshot
    # (see module-level ``_SKIP_ATTRS`` which excludes "phase" from
    # the averaging loop above).
    if p_i_snap is not None:
        setattr(dst, "phase", p_i_snap)


# Extensive vertex quantities: conserved exactly by split/collapse.
_EXTENSIVE_ATTRS = ("m", "m_phase")


def _one_ring_area(v) -> float:
    """Total unsigned area of all triangles incident to ``v``.

    With barycentric duals, ``v``'s dual-cell area is exactly one third
    of this value; callers only use *ratios* of these areas, so the
    factor 1/3 is dropped.
    """
    total = 0.0
    seen: set = set()
    for v_a in v.nn:
        for v_b in v_a.nn:
            if v_b is v or v_b not in v.nn:
                continue
            key = ((id(v_a), id(v_b)) if id(v_a) < id(v_b)
                   else (id(v_b), id(v_a)))
            if key in seen:
                continue
            seen.add(key)
            total += abs(triangle_area(v, v_a, v_b))
    return total


def _transfer_extensive(v_m, v_i, v_j, f_i: float, f_j: float) -> tuple:
    """Move fractions ``f_i`` / ``f_j`` of the endpoints' extensive
    quantities (``m``, ``m_phase``) onto the new midpoint ``v_m``.

    The transferred amount is SUBTRACTED from the endpoints, so the
    total over the mesh is invariant to round-off.

    Returns the two scalar mass transfers ``(dm_i, dm_j)`` for the
    ``"m"`` attribute (``None`` where the endpoint has no usable mass);
    callers use them for momentum-conserving velocity mixing.
    """
    dm_scalar = {id(v_i): None, id(v_j): None}
    for attr in _EXTENSIVE_ATTRS:
        taken = None
        for v_src, f in ((v_i, f_i), (v_j, f_j)):
            val = getattr(v_src, attr, None)
            if val is None:
                continue
            try:
                if isinstance(val, np.ndarray):
                    val = val.astype(float)
                else:
                    val = float(val)
            except (TypeError, ValueError):
                continue
            dm = f * val
            setattr(v_src, attr, val - dm)
            taken = dm if taken is None else taken + dm
            if attr == "m":
                dm_scalar[id(v_src)] = float(dm)
        if taken is not None:
            setattr(v_m, attr, taken)
    return dm_scalar[id(v_i)], dm_scalar[id(v_j)]


# ---------------------------------------------------------------------------
# Triangle enumeration around an edge
# ---------------------------------------------------------------------------

def triangles_around_edge(v_i, v_j) -> list:
    """Return the list of *opposite* vertices for the triangles sharing
    edge ``(v_i, v_j)``.

    Each entry ``v_k`` represents the triangle ``(v_i, v_j, v_k)``.
    For a manifold interior edge this list has length 2; for a boundary
    edge it has length 1.  For non-manifold or degenerate meshes it may
    have length 0 or >2.
    """
    return [v for v in (v_i.nn & v_j.nn) if v is not v_i and v is not v_j]


# ---------------------------------------------------------------------------
# Edge split
# ---------------------------------------------------------------------------

def edge_split_2d(HC, v_i, v_j) -> Optional[Any]:
    """Split edge ``(v_i, v_j)`` by inserting its midpoint.

    The two adjacent triangles are replaced by four.  The new midpoint
    vertex inherits averaged field values from ``v_i`` and ``v_j``.

    Returns the new vertex, or ``None`` if the split cannot be performed
    (e.g. midpoint coordinate already present in the cache).
    """
    if v_j not in v_i.nn:
        return None

    # 1. Opposite vertices before topology change
    opposites = list(triangles_around_edge(v_i, v_j))

    # Pre-split geometry for the mass-conservative transfer below —
    # must be measured BEFORE the connectivity changes.
    area_T = sum(abs(triangle_area(v_i, v_j, v_k)) for v_k in opposites)
    ring_i = _one_ring_area(v_i)
    ring_j = _one_ring_area(v_j)

    # 2. Midpoint coordinate
    x_m = tuple(0.5 * (np.asarray(v_i.x, dtype=float)
                       + np.asarray(v_j.x, dtype=float)))

    # If the midpoint already exists in the cache, abort — we refuse to
    # re-use an unrelated vertex for this operation.
    if x_m in HC.V.cache:
        return None

    v_m = HC.V[x_m]  # Creates and indexes the new vertex

    # 3. Connect the new midpoint to v_i, v_j, and both opposites.
    #    Then disconnect the original edge (v_i, v_j).
    v_m.connect(v_i)
    v_m.connect(v_j)
    for v_k in opposites:
        v_m.connect(v_k)

    v_i.disconnect(v_j)

    # 4. Propagate fields (velocity, pressure, mass, phase, ...).
    _carry_attrs(v_m, v_i, v_j)

    # Mass is EXTENSIVE — conserve it exactly.  With barycentric duals
    # the midpoint's new dual cell has area (1/3)(|T_k| + |T_l|),
    # carved half out of v_i's cell and half out of v_j's (the opposite
    # vertices' cells are unchanged by the split).  Each endpoint
    # therefore cedes dA = (1/6)(|T_k| + |T_l|) of its pre-split dual
    # area A = ring/3, i.e. the mass fraction
    #     f = dA / A = (|T_k| + |T_l|) / (2 * ring).
    # Transferring f*m FROM each endpoint keeps sum(m) invariant and
    # leaves a uniform density field exactly uniform.  (Previously the
    # midpoint was ASSIGNED the arithmetic mean 0.5*(m_i + m_j) out of
    # thin air, inflating total mass on every split — documented
    # blow-up: mass 9.7 -> 187 over 100 steps under adaptive remesh.)
    # The same fractions are applied to per-phase masses (``m_phase``)
    # so v.m == sum(v.m_phase) stays consistent.
    f_i = min(0.5 * area_T / ring_i, 1.0) if ring_i > 0.0 else 0.0
    f_j = min(0.5 * area_T / ring_j, 1.0) if ring_j > 0.0 else 0.0
    dm_i, dm_j = _transfer_extensive(v_m, v_i, v_j, f_i, f_j)

    # Momentum-conserving midpoint velocity: the transferred parcels
    # carry their source velocities, mixed mass-weighted at the
    # midpoint.  This keeps sum(m*u) exactly invariant and is strictly
    # KE-non-increasing (the plain arithmetic average from _carry_attrs
    # injects momentum/energy whenever the transfer is asymmetric).
    # Falls back to the _carry_attrs average when masses are missing.
    if dm_i is not None and dm_j is not None and (dm_i + dm_j) > 0.0:
        u_i = getattr(v_i, "u", None)
        u_j = getattr(v_j, "u", None)
        if (isinstance(u_i, np.ndarray) and isinstance(u_j, np.ndarray)
                and u_i.shape == u_j.shape):
            v_m.u = (dm_i * u_i.astype(float)
                     + dm_j * u_j.astype(float)) / (dm_i + dm_j)

    # Boundary inheritance: a midpoint of two boundary vertices that
    # shared a boundary edge (only one opposite triangle) is itself a
    # boundary vertex.
    is_bdry = (getattr(v_i, "boundary", False)
               and getattr(v_j, "boundary", False)
               and len(opposites) <= 1)
    v_m.boundary = bool(is_bdry)

    # Incremental simplicial-representation update: each triangle
    # (v_i, v_j, v_k) is replaced by (v_i, v_m, v_k) and (v_j, v_m, v_k).
    if getattr(HC, "_SC", None) is not None:
        old = [(v_i, v_j, v_k) for v_k in opposites]
        new = []
        for v_k in opposites:
            new.append((v_i, v_m, v_k))
            new.append((v_j, v_m, v_k))
        HC._sc_notify("remove_simplices", simplices=old)
        HC._sc_notify("add_simplices", simplices=new)

    return v_m


# ---------------------------------------------------------------------------
# Edge collapse
# ---------------------------------------------------------------------------

def _would_invert(v_keep, v_remove) -> bool:
    """Detect whether collapsing ``v_remove -> v_keep`` would flip any
    triangle orientation in ``v_remove``'s 1-ring.

    We walk every triangle ``(v_remove, v_a, v_b)`` in the current mesh
    and compute the signed area; then we compute the signed area of
    the replacement triangle ``(v_keep, v_a, v_b)``.  If any pair has
    opposite signs (and the original area was non-zero), the collapse
    would invert a triangle.
    """
    p_keep = np.asarray(v_keep.x_a, dtype=float)
    p_remove = np.asarray(v_remove.x_a, dtype=float)

    # Build triangles around v_remove from its 1-ring.
    nbrs = [v for v in v_remove.nn if v is not v_keep]
    checked: set = set()
    for v_a in nbrs:
        for v_b in v_a.nn:
            if v_b is v_remove or v_b is v_keep or v_b is v_a:
                continue
            if v_b not in v_remove.nn:
                continue
            # Triangle (v_remove, v_a, v_b) — canonical ordering to dedupe
            key = tuple(sorted((id(v_a), id(v_b))))
            if key in checked:
                continue
            checked.add(key)

            p_a = np.asarray(v_a.x_a, dtype=float)[:2]
            p_b = np.asarray(v_b.x_a, dtype=float)[:2]
            e1_old = p_a - p_remove[:2]
            e2_old = p_b - p_remove[:2]
            s_old = e1_old[0] * e2_old[1] - e1_old[1] * e2_old[0]

            # Triangle (v_remove, v_a, v_b) becomes (v_keep, v_a, v_b)
            # — but only if (v_keep, v_a, v_b) isn't already a triangle
            # (otherwise it collapses, not flips).
            if v_a in v_keep.nn and v_b in v_keep.nn and v_a in v_b.nn:
                # The collapsed triangle disappears (merges with the
                # other one sharing edge (v_a, v_b)); that's expected
                # only for the two triangles adjacent to edge
                # (v_keep, v_remove).
                continue

            e1_new = p_a - p_keep[:2]
            e2_new = p_b - p_keep[:2]
            s_new = e1_new[0] * e2_new[1] - e1_new[1] * e2_new[0]

            if abs(s_old) < 1e-14:
                continue
            if s_old * s_new <= 0.0:
                return True
    return False


def edge_collapse_2d(HC, v_i, v_j) -> bool:
    """Collapse edge ``(v_i, v_j)`` by removing ``v_j`` and rewiring its
    neighbourhood to ``v_i``.

    The surviving vertex is placed at the midpoint of ``v_i`` and
    ``v_j``.  Callers must call :func:`hyperct.remesh.can_collapse`
    first: this function assumes both endpoints are interior
    (non-boundary, non-cross-phase).

    Returns True on success, False otherwise (disconnected edge,
    would invert a triangle, or the cache already contains a vertex
    at the midpoint position).
    """
    if v_j not in v_i.nn:
        return False

    # Reject if the move would invert adjacent triangles.
    if _would_invert(v_i, v_j):
        return False
    if _would_invert(v_j, v_i):
        return False

    # Midpoint merge — both endpoints are interior by precondition.
    x_new = tuple(0.5 * (np.asarray(v_i.x, dtype=float)
                         + np.asarray(v_j.x, dtype=float)))

    # Abort BEFORE mutating anything when the merged position is
    # already occupied by a third vertex, or when v_j is stale (not
    # the vertex stored at its own coordinates, so HC.V.remove would
    # fail).  Previously the position collision was only detected
    # after the connectivity merge and the function returned True with
    # the survivor left at its old position (upstream bug list,
    # docs_temp/code_map/hyperct_upstream.md).
    occupant = HC.V.cache.get(x_new)
    if occupant is not None and occupant is not v_i and occupant is not v_j:
        return False
    if HC.V.cache.get(v_j.x) is not v_j:
        return False

    # Snapshot masses/velocities for the momentum-conserving merge
    # below (before _carry_attrs overwrites v_i.u with the plain
    # average).
    _m_i = getattr(v_i, "m", None)
    _m_j = getattr(v_j, "m", None)
    _u_i = getattr(v_i, "u", None)
    _u_j = getattr(v_j, "u", None)

    # Average simulation fields into v_i.
    _carry_attrs(v_i, v_i, v_j)
    # Extensive quantities are ADDITIVE on collapse (two parcels merge
    # into one): total mass AND per-phase masses.  The removed vertex's
    # share must be redistributed, never dropped — ``m_phase`` used to
    # fall through the averaging branch of _carry_attrs, silently
    # destroying half of (m_phase_i + m_phase_j).
    for _attr in _EXTENSIVE_ATTRS:
        val_j = getattr(v_j, _attr, None)
        if val_j is None:
            continue
        val_i = getattr(v_i, _attr, None)
        try:
            if isinstance(val_j, np.ndarray) or isinstance(val_i, np.ndarray):
                base = (np.asarray(val_i, dtype=float)
                        if val_i is not None else 0.0)
                setattr(v_i, _attr, base + np.asarray(val_j, dtype=float))
            else:
                base = float(val_i) if val_i is not None else 0.0
                setattr(v_i, _attr, base + float(val_j))
        except (TypeError, ValueError):
            pass

    # Momentum-conserving merged velocity (inelastic merge of two
    # parcels): u = (m_i*u_i + m_j*u_j) / (m_i + m_j).  Keeps
    # sum(m*u) exactly invariant and is KE-non-increasing; the plain
    # average written by _carry_attrs above injects momentum whenever
    # the masses differ.  Falls back to that average when masses or
    # velocities are unavailable.
    try:
        if (_m_i is not None and _m_j is not None
                and isinstance(_u_i, np.ndarray)
                and isinstance(_u_j, np.ndarray)
                and _u_i.shape == _u_j.shape):
            m_sum = float(_m_i) + float(_m_j)
            if m_sum > 0.0:
                v_i.u = (float(_m_i) * _u_i.astype(float)
                         + float(_m_j) * _u_j.astype(float)) / m_sum
    except (TypeError, ValueError):
        pass

    # Reconnect v_j's neighbours to v_i (skip v_i itself and vertices
    # already connected to v_i).
    for nb in list(v_j.nn):
        if nb is v_i:
            continue
        v_i.connect(nb)

    # Remove v_j from the cache (also disconnects it from its neighbours).
    try:
        HC.V.remove(v_j)
    except KeyError:
        return False

    # A collapse rewires v_j's incident triangles onto v_i (non-local); the
    # incremental vertex-removal hook only drops v_j's simplices, so mark the
    # representation dirty for a full re-derivation on next read.
    if getattr(HC, "_SC", None) is not None:
        HC._sc_notify("collapse")

    # Move v_i to the merged position, if it changed.
    if tuple(v_i.x) != x_new:
        if x_new in HC.V.cache and HC.V.cache[x_new] is not v_i:
            # Unreachable in practice: collisions are rejected up front
            # before any mutation.  Kept as a defensive guard — at this
            # point the collapse has already happened, so we still
            # report success rather than lie about the topology change.
            return True
        HC.V.move(v_i, x_new)

    return True


# ---------------------------------------------------------------------------
# Edge flip
# ---------------------------------------------------------------------------

def edge_flip_2d(HC, v_i, v_j, min_quality_gain: float = 0.0) -> bool:
    """Flip edge ``(v_i, v_j)`` iff it has exactly two adjacent triangles
    ``(v_i, v_j, v_k)`` and ``(v_i, v_j, v_l)``, and flipping improves
    the minimum-angle quality.

    After the flip, the new edge is ``(v_k, v_l)`` and the triangles
    become ``(v_k, v_l, v_i)`` and ``(v_k, v_l, v_j)``.

    Parameters
    ----------
    HC : Complex
        The simplicial complex owning the vertices.  Used to keep an
        active simplicial representation (``HC.SC``) in sync via an
        incremental simplex swap; otherwise the flip operates purely on
        the vertex connectivity (``v.nn``).
    v_i, v_j : vertex
        The two endpoints of the edge to flip.
    min_quality_gain : float
        Only perform the flip when
        ``min(angle_new) - min(angle_old) > min_quality_gain`` (radians).
        Set to 0 for pure Delaunay flips.

    Returns True on success, False otherwise.
    """
    if v_j not in v_i.nn:
        return False

    opposites = triangles_around_edge(v_i, v_j)
    if len(opposites) != 2:
        return False
    v_k, v_l = opposites

    # The new edge (v_k, v_l) must not already exist — otherwise the
    # flip would create a duplicate edge / degenerate simplex.
    if v_l in v_k.nn:
        return False

    # Quality check.
    old_q = min(triangle_min_angle(v_i, v_j, v_k),
                triangle_min_angle(v_i, v_j, v_l))
    new_q = min(triangle_min_angle(v_k, v_l, v_i),
                triangle_min_angle(v_k, v_l, v_j))
    if new_q - old_q <= min_quality_gain:
        return False

    # Orientation check: the two new triangles must both have positive
    # signed area on the same side as the originals (prevents flipping
    # into a concave quadrilateral).
    s_old = triangle_area(v_i, v_j, v_k) + triangle_area(v_i, v_j, v_l)
    s_new = triangle_area(v_k, v_l, v_i) + triangle_area(v_k, v_l, v_j)
    # For a convex quadrilateral the sum of signed areas is the same
    # (it's the area of the quadrilateral).  If signs differ the flip
    # would create a bowtie.
    if abs(s_old) > 1e-14 and s_old * s_new < 0.0:
        return False
    # And individually non-degenerate:
    if triangle_min_angle(v_k, v_l, v_i) <= 0.0:
        return False
    if triangle_min_angle(v_k, v_l, v_j) <= 0.0:
        return False

    # Pre-flip geometry for the mass-conservative transfer below.
    # Barycentric dual shares of the flip quad (each triangle donates
    # |T|/3 to each of its corners):
    #   old: a_i = a_j = (t_k + t_l)/3,  a_k = t_k/3,  a_l = t_l/3
    #   new: a_k = a_l = (t_i + t_j)/3,  a_i = t_i/3,  a_j = t_j/3
    # with t_i + t_j = t_k + t_l = A_quad, so v_i/v_j ALWAYS lose dual
    # area and v_k/v_l ALWAYS gain.  Without a matching mass transfer a
    # flip steps the density of all four vertices by O(1) — the same
    # EOS-noise mechanism the split/collapse fixes closed
    # (lane4-remesh-upstream 2026-07-02; observed as a contact-corner
    # density halving in the capillary-rise dynCA case).
    t_k = abs(triangle_area(v_i, v_j, v_k))
    t_l = abs(triangle_area(v_i, v_j, v_l))
    t_i = abs(triangle_area(v_k, v_l, v_i))
    t_j = abs(triangle_area(v_k, v_l, v_j))
    A_quad = t_k + t_l
    ring_i = _one_ring_area(v_i)
    ring_j = _one_ring_area(v_j)

    # Perform the flip: remove (v_i, v_j), add (v_k, v_l).
    v_i.disconnect(v_j)
    v_k.connect(v_l)

    # Mass transfer: donors v_i, v_j give the fraction of their mass
    # matching their dual-area loss (at their own density); the pool is
    # distributed to v_k, v_l pro rata of dual-area gain.  Conserves
    # sum(m) (and sum(m_phase)) exactly and leaves a uniform density
    # field exactly uniform.  Velocities of the gainers are mixed
    # mass-weighted with the donated parcel, keeping sum(m*u) invariant.
    gain_k = (A_quad - t_k) / 3.0
    gain_l = (A_quad - t_l) / 3.0
    gain_tot = gain_k + gain_l
    if gain_tot > 0.0 and A_quad > 0.0:
        f_i = min((A_quad - t_i) / max(ring_i, 1e-300), 1.0)
        f_j = min((A_quad - t_j) / max(ring_j, 1e-300), 1.0)
        pooled = {}
        dm_pool = 0.0
        u_mom = None
        for v_src, f in ((v_i, f_i), (v_j, f_j)):
            if f <= 0.0:
                continue
            for attr in _EXTENSIVE_ATTRS:
                val = getattr(v_src, attr, None)
                if val is None:
                    continue
                try:
                    if isinstance(val, np.ndarray):
                        val = val.astype(float)
                    else:
                        val = float(val)
                except (TypeError, ValueError):
                    continue
                dm = f * val
                setattr(v_src, attr, val - dm)
                pooled[attr] = dm if attr not in pooled else pooled[attr] + dm
                if attr == "m":
                    dm_pool += float(dm)
                    u_src = getattr(v_src, "u", None)
                    if isinstance(u_src, np.ndarray):
                        mom = float(dm) * u_src.astype(float)
                        u_mom = mom if u_mom is None else u_mom + mom
        for v_dst, w in ((v_k, gain_k / gain_tot), (v_l, gain_l / gain_tot)):
            for attr, dm in pooled.items():
                cur = getattr(v_dst, attr, None)
                add = w * dm
                if cur is None:
                    setattr(v_dst, attr, add)
                    continue
                try:
                    if isinstance(cur, np.ndarray):
                        cur = cur.astype(float)
                    else:
                        cur = float(cur)
                except (TypeError, ValueError):
                    continue
                setattr(v_dst, attr, cur + add)
            # momentum-conserving velocity mix for the received parcel
            if dm_pool > 0.0 and u_mom is not None:
                m_new = getattr(v_dst, "m", None)
                u_dst = getattr(v_dst, "u", None)
                if (m_new is not None and isinstance(u_dst, np.ndarray)
                        and float(m_new) > 0.0):
                    dm_w = w * dm_pool
                    m_old = float(m_new) - dm_w
                    v_dst.u = (m_old * u_dst.astype(float)
                               + dm_w * (u_mom / dm_pool)) / float(m_new)

    # Incremental simplicial-representation update: the two triangles
    # (v_i, v_j, v_k) and (v_i, v_j, v_l) become (v_k, v_l, v_i) and
    # (v_k, v_l, v_j).
    if getattr(HC, "_SC", None) is not None:
        HC._sc_notify("remove_simplices",
                      simplices=[(v_i, v_j, v_k), (v_i, v_j, v_l)])
        HC._sc_notify("add_simplices",
                      simplices=[(v_k, v_l, v_i), (v_k, v_l, v_j)])
    return True
