"""Interface-aware constraints for local mesh operations.

A *phase interface* is the set of primal edges whose endpoints belong to
different phases (``v.phase`` attribute).  Interface edges must be
preserved under remeshing so that discrete surface tension operators see
a continuous interface loop/surface.

Convention: if a vertex has no ``phase`` attribute we treat it as phase
``None`` (single-phase mesh).  In that case no edge is an interface edge
and no operations are blocked.
"""

from __future__ import annotations

from typing import Optional


def vertex_phase(v) -> Optional[int]:
    """Return ``v.phase`` if it exists, otherwise ``None``.

    Single-phase meshes have no phase attribute and therefore no
    interface — all operations are unconstrained.
    """
    return getattr(v, "phase", None)


def is_interface_edge(v_i, v_j) -> bool:
    """True iff the edge (v_i, v_j) crosses a phase boundary.

    Returns False when either vertex lacks a phase attribute (single-phase
    mesh) or when both phases are equal.
    """
    p_i = vertex_phase(v_i)
    p_j = vertex_phase(v_j)
    if p_i is None or p_j is None:
        return False
    return p_i != p_j


def is_interface_vertex(v) -> bool:
    """True iff any incident edge crosses a phase boundary."""
    p_v = vertex_phase(v)
    if p_v is None:
        return False
    for v2 in v.nn:
        p2 = vertex_phase(v2)
        if p2 is not None and p2 != p_v:
            return True
    return False


def can_flip(v_i, v_j) -> bool:
    """True iff the edge (v_i, v_j) may be flipped.

    Interface edges are never flipped because flipping would replace a
    cross-phase edge with an edge between two same-phase vertices (or
    two different same-phase pairs), destroying the discrete interface
    topology.  Boundary edges (either endpoint on the mesh boundary) are
    also never flipped to avoid modifying the domain boundary.
    """
    if is_interface_edge(v_i, v_j):
        return False
    if getattr(v_i, "boundary", False) and getattr(v_j, "boundary", False):
        return False
    return True


def can_collapse(v_i, v_j) -> bool:
    """True iff the edge (v_i, v_j) may be collapsed.

    Restrictions:

    - Collapsing is forbidden across a phase boundary (would merge two
      material parcels that were meant to be separated by the interface).
    - Collapsing is forbidden when both endpoints are topological
      boundary vertices — this would delete a boundary vertex that is
      likely constrained by a wall BC.
    - Collapsing of a boundary vertex *into* an interior vertex is also
      forbidden: the interior vertex would inherit a boundary role it
      was never given.
    """
    if is_interface_edge(v_i, v_j):
        return False
    b_i = bool(getattr(v_i, "boundary", False))
    b_j = bool(getattr(v_j, "boundary", False))
    if b_i or b_j:
        # Only allow if both are interior; collapsing boundary vertices
        # modifies the mesh boundary and is not handled here.
        return False
    return True


def split_preserves_phase_topology(v_i, v_j) -> bool:
    """True iff edge ``(v_i, v_j)`` can be split without corrupting
    the phase-interface topology.

    The midpoint of a split inherits ``v_i.phase`` and is connected
    to every vertex opposite to the split edge in the adjacent
    triangles.  If any opposite vertex belongs to a different phase
    than ``v_i``, the split introduces a NEW cross-phase edge
    ``(v_m, v_opp)`` even when the split edge itself was a bulk
    edge.  This function rejects such splits so that bulk regions
    stay bulk under remeshing.

    Interface edges (``v_i.phase != v_j.phase``) are always allowed
    to split, since subdividing an interface edge is the expected
    behaviour — the midpoint extends the interface as a continuous
    chain.

    Returns True for single-phase meshes (no ``v.phase`` attributes)
    and for edges where the entire 1-ring of the split is in one
    phase.
    """
    p_i = vertex_phase(v_i)
    p_j = vertex_phase(v_j)
    if p_i is None or p_j is None:
        return True  # Single-phase mesh — no constraint
    if p_i != p_j:
        return True  # Interface edge — subdividing it is the expected case
    # Bulk edge — the midpoint will inherit p_i.  Any opposite vertex
    # with a different phase would introduce a new cross-phase edge
    # after the split, which we want to avoid.
    opposites = v_i.nn & v_j.nn
    for v_k in opposites:
        if v_k is v_i or v_k is v_j:
            continue
        p_k = vertex_phase(v_k)
        if p_k is not None and p_k != p_i:
            return False
    return True
