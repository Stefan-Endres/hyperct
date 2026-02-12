"""
Disconnectivity graph visualization for energy landscapes.

Builds tree representations of energy landscape connectivity from
minima and transition states, and visualizes them as disconnectivity
graphs (also known as "palm tree" or "barrier tree" diagrams).

Licensing note:

The code in this file is derived from heavily modified code (12.01.2023)
from the Pele repository https://github.com/pele-python/pele, released
under GPLv3 licence by Jacob Stevenson and other listed contributors.

sqlalchemy and networkx dependencies have been removed. The graph
representation uses a lightweight SimpleGraph class internal to hyperct.

Original license:

pele is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pele is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pele.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import deque
import operator

import numpy as np

__all__ = ["DisconnectivityGraph", "Minimum", "TransitionState",
           "SimpleGraph", "database_from_complex"]


# ---------------------------------------------------------------------------
# Lightweight graph (replaces networkx)
# ---------------------------------------------------------------------------

class SimpleGraph:
    """Minimal undirected graph with edge attributes.

    Replaces ``networkx.Graph`` for the subset of operations used by
    ``DisconnectivityGraph``.  Nodes are any hashable objects; each edge
    can carry an arbitrary attribute dict.
    """

    def __init__(self):
        self._adj: dict[object, dict[object, dict]] = {}

    # -- node operations ----------------------------------------------------

    def add_node(self, n):
        if n not in self._adj:
            self._adj[n] = {}

    def add_nodes_from(self, nodes):
        for n in nodes:
            self.add_node(n)

    def remove_node(self, n):
        for nbr in list(self._adj.get(n, {})):
            del self._adj[nbr][n]
        del self._adj[n]

    def nodes(self):
        return list(self._adj)

    def number_of_nodes(self):
        return len(self._adj)

    def degree(self, n):
        return len(self._adj[n])

    # -- edge operations ----------------------------------------------------

    def add_edge(self, u, v, **attr):
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = attr
        self._adj[v][u] = attr

    def add_edges_from(self, edge_list):
        for item in edge_list:
            u, v = item[0], item[1]
            attr = item[2] if len(item) > 2 else {}
            self.add_edge(u, v, **attr)

    def remove_edge(self, u, v):
        del self._adj[u][v]
        del self._adj[v][u]

    def edges(self):
        seen = set()
        result = []
        for u, nbrs in self._adj.items():
            for v in nbrs:
                key = (id(u), id(v)) if id(u) < id(v) else (id(v), id(u))
                if key not in seen:
                    seen.add(key)
                    result.append((u, v))
        return result

    def number_of_edges(self):
        return len(self.edges())

    def get_edge_data(self, u, v):
        return self._adj[u].get(v, None)

    def get_edge_attributes(self, attr_name):
        """Return dict mapping ``(u, v)`` to the attribute value."""
        result = {}
        for u, v in self.edges():
            data = self._adj[u][v]
            if attr_name in data:
                result[(u, v)] = data[attr_name]
        return result

    # -- connectivity -------------------------------------------------------

    def _bfs_component(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for nbr in self._adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return visited

    def node_connected_component(self, node):
        return self._bfs_component(node)

    def connected_components(self):
        visited = set()
        components = []
        for n in self._adj:
            if n not in visited:
                comp = self._bfs_component(n)
                visited |= comp
                components.append(comp)
        return components

    # -- subgraph -----------------------------------------------------------

    def subgraph(self, nodes):
        node_set = set(nodes)
        sg = SimpleGraph()
        for n in node_set:
            sg.add_node(n)
        for u, v in self.edges():
            if u in node_set and v in node_set:
                sg.add_edge(u, v, **self._adj[u][v])
        return sg

    def copy(self):
        g = SimpleGraph()
        for n in self._adj:
            g.add_node(n)
        for u, v in self.edges():
            g.add_edge(u, v, **self._adj[u][v])
        return g


# ---------------------------------------------------------------------------
# Union-Find (replaces networkx.utils.UnionFind)
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self):
        self.parents: dict = {}
        self._weights: dict = {}

    def __getitem__(self, obj):
        """Find the root of *obj*, creating a singleton set if needed."""
        if obj not in self.parents:
            self.parents[obj] = obj
            self._weights[obj] = 1
            return obj

        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # path compression
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, u, v):
        """Merge the sets containing *u* and *v*."""
        root_u = self[u]
        root_v = self[v]
        if root_u == root_v:
            return

        # union by weight
        if self._weights[root_u] < self._weights[root_v]:
            self.parents[root_u] = root_v
            self._weights[root_v] += self._weights[root_u]
        else:
            self.parents[root_v] = root_u
            self._weights[root_u] += self._weights[root_v]

    def groups_iter(self):
        """Iterate over current root elements."""
        return (c for c, c1 in self.parents.items() if c == c1)


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

class Tree:
    """Generic tree node with parent/children navigation.

    Each instance is a node that can have many children but only one parent.
    A node with no parent is a root; a node with no children is a leaf.
    """

    def __init__(self, parent=None):
        self.subtrees: list["Tree"] = []
        self.data: dict = {}
        self.parent: "Tree | None" = parent
        if parent is not None:
            parent.add_branch(self)

    def add_branch(self, branch: "Tree"):
        self.subtrees.append(branch)
        branch.parent = self

    def make_branch(self) -> "Tree":
        return self.__class__(parent=self)

    def get_branches(self) -> list["Tree"]:
        return self.subtrees

    def number_of_branches(self) -> int:
        return len(self.subtrees)

    def is_leaf(self) -> bool:
        return len(self.subtrees) == 0

    def number_of_leaves(self) -> int:
        if self.is_leaf():
            return 1
        return sum(t.number_of_leaves() for t in self.subtrees)

    def get_leaves(self) -> list["Tree"]:
        if self.is_leaf():
            return [self]
        leaves = []
        for t in self.subtrees:
            leaves += t.get_leaves()
        return leaves

    def leaf_iterator(self):
        if self.is_leaf():
            yield self
        else:
            for t in self.subtrees:
                yield from t.leaf_iterator()

    def get_all_trees(self):
        yield self
        for branch in self.subtrees:
            yield from branch.get_all_trees()

    def number_of_subtrees(self) -> int:
        return 1 + sum(b.number_of_subtrees() for b in self.subtrees)

    def get_ancestors(self):
        if self.parent is not None:
            yield self.parent
            yield from self.parent.get_ancestors()


class TreeLeastCommonAncestor:
    """Find the least common ancestor of a set of tree nodes."""

    def __init__(self, trees):
        self.start_trees = trees
        self.run()

    def run(self):
        common_ancestors: set = set()
        for tree in self.start_trees:
            parents = set(tree.get_ancestors())
            parents.add(tree)
            if not common_ancestors:
                common_ancestors.update(parents)
            else:
                common_ancestors.intersection_update(parents)
                assert common_ancestors

        if not common_ancestors:
            raise ValueError("the trees have no common ancestors")

        common_ancestors_list = sorted(
            common_ancestors,
            key=lambda t: len(list(t.get_ancestors())),
        )
        self.least_common_ancestor = common_ancestors_list[-1]
        return self.least_common_ancestor

    def get_all_paths_to_common_ancestor(self):
        trees = set(self.start_trees)
        for tree in self.start_trees:
            for parent in tree.get_ancestors():
                trees.add(parent)
                if parent == self.least_common_ancestor:
                    break
        return trees


class DGTree(Tree):
    """Tree node specialised for disconnectivity graphs."""

    def contains_minimum(self, min1) -> bool:
        return any(leaf.data["minimum"] == min1 for leaf in self.get_leaves())

    def get_minima(self):
        return [leaf.data["minimum"] for leaf in self.get_leaves()]

    def get_one_minimum(self):
        if self.is_leaf():
            return self.data["minimum"]
        key = "_random_minimum"
        if key in self.data:
            return self.data[key]
        m = self.get_branches()[0].get_one_minimum()
        self.data[key] = m
        return m


# ---------------------------------------------------------------------------
# Tree builder (Kruskal-style algorithm)
# ---------------------------------------------------------------------------

class _MakeTree:
    """Build the disconnectivity tree from minima, transition states, and
    energy levels.

    The algorithm adds transition states in ascending energy order and
    records connectivity snapshots at each energy level — analogous to
    Kruskal's minimum spanning tree algorithm.
    """

    def __init__(self, minima, transition_states, energy_levels, get_energy=None):
        self.minima = minima
        self.transition_states = transition_states
        self.energy_levels = energy_levels
        self._get_energy = get_energy
        self.union_find = UnionFind()
        self.minimum_to_leave: dict = {}

    def get_energy(self, ts):
        if self._get_energy is None:
            return ts.energy
        return self._get_energy(ts)

    def _new_leaf(self, m):
        leaf = DGTree()
        leaf.data["minimum"] = m
        self.minimum_to_leave[m] = leaf
        return leaf

    def make_tree(self):
        tslist = [ts for ts in self.transition_states if ts.minimum1 != ts.minimum2]
        tslist = list(set(tslist))
        tslist.sort(key=lambda ts: -self.get_energy(ts))
        self.transition_states = tslist

        trees: list = []
        for ilevel in range(len(self.energy_levels)):
            trees = self._do_next_level(ilevel, trees)

        energy_levels = self.energy_levels
        if len(trees) == 1:
            self.tree = trees[0]
        else:
            self.tree = DGTree()
            for t in trees:
                self.tree.add_branch(t)
            self.tree.data["ilevel"] = len(energy_levels) - 1
            de = energy_levels[-1] - energy_levels[-2]
            self.tree.data["ethresh"] = energy_levels[-1] + de
            self.tree.data["children_not_connected"] = True

        return self.tree

    def _add_edge(self, min1, min2):
        new_minima = []
        if min1 not in self.union_find.parents:
            new_minima.append(min1)
        if min2 not in self.union_find.parents:
            new_minima.append(min2)
        self.union_find.union(min1, min2)
        return new_minima

    def _do_next_level(self, ilevel, previous_trees):
        ethresh = self.energy_levels[ilevel]

        tslist = self.transition_states
        while tslist:
            ts = tslist[-1]
            if self.get_energy(ts) >= ethresh:
                break
            new_minima = self._add_edge(ts.minimum1, ts.minimum2)
            for m in new_minima:
                previous_trees.append(self._new_leaf(m))
            tslist.pop()

        newtrees = []
        color_to_tree: dict = {}
        for c in self.union_find.groups_iter():
            newtree = DGTree()
            newtree.data["ilevel"] = ilevel
            newtree.data["ethresh"] = ethresh
            newtrees.append(newtree)
            color_to_tree[c] = newtree

        for tree in previous_trees:
            m = tree.get_one_minimum()
            c = self.union_find[m]
            parent = color_to_tree[c]
            if tree.number_of_branches() == 1:
                subtree = next(iter(tree.subtrees))
                parent.add_branch(subtree)
            else:
                parent.add_branch(tree)

        return newtrees


# ---------------------------------------------------------------------------
# Colouring helpers
# ---------------------------------------------------------------------------

class ColorDGraphByGroups:
    """Colour tree nodes by grouping of minima."""

    def __init__(self, tree_graph, groups, colors=None):
        self.tree_graph = tree_graph
        self._minimum_to_color: dict = {}
        self.color_list = self._resolve_colors(len(groups), colors)
        for color, group in zip(self.color_list, groups):
            for minimum in group:
                self._minimum_to_color[minimum] = color
        self._tree_to_colors: dict = {}

    @staticmethod
    def _resolve_colors(number, colors=None):
        if colors is not None:
            from matplotlib.colors import ColorConverter
            cc = ColorConverter()
            if number != len(colors):
                raise ValueError("len(colors) must equal number of groups")
            return [cc.to_rgb(c) for c in colors]
        from matplotlib import colormaps
        cmap = colormaps.get_cmap("Dark2")
        return [cmap(i) for i in np.linspace(0., 1., number)]

    def _minimum_color(self, minimum):
        return self._minimum_to_color.get(minimum)

    def tree_get_colors(self, tree):
        if tree in self._tree_to_colors:
            return self._tree_to_colors[tree]
        if tree.is_leaf():
            color = self._minimum_color(tree.data["minimum"])
            colors = frozenset([color]) if color is not None else None
        else:
            sub_colors = [self.tree_get_colors(st) for st in tree.get_branches()]
            if None in sub_colors:
                colors = None
            else:
                colors = frozenset(c for cs in sub_colors for c in cs)
        self._tree_to_colors[tree] = colors
        return colors

    def _pick_color(self, colors):
        for color in reversed(self.color_list):
            if color in colors:
                return color

    def run(self):
        for tree in self.tree_graph.get_all_trees():
            colors = self.tree_get_colors(tree)
            if colors is not None:
                tree.data["colour"] = self._pick_color(colors)


class ColorDGraphByValue:
    """Colour tree nodes by a scalar value associated with each minimum."""

    def __init__(self, tree_graph, minimum_to_value, colormap=None,
                 normalize_values=True):
        self.tree_graph = tree_graph
        self.minimum_to_value = minimum_to_value
        if colormap is None:
            from matplotlib import colormaps
            self.colormap = colormaps.get_cmap("winter")
        else:
            self.colormap = colormap

        self._tree_to_value: dict = {}

        if normalize_values:
            values = [self.minimum_to_value(leaf.data["minimum"])
                      for leaf in self.tree_graph.leaf_iterator()]
            values = [v for v in values if v is not None]
            self.maxval = max(values)
            self.minval = min(values)
        else:
            self.minval = None
            self.maxval = None

    def value_to_color(self, value):
        if self.minval is None:
            vnorm = value
        else:
            vnorm = (value - self.minval) / (self.maxval - self.minval)
        return self.colormap(vnorm)

    def tree_get_value(self, tree):
        if tree in self._tree_to_value:
            return self._tree_to_value[tree]
        if tree.is_leaf():
            value = self.minimum_to_value(tree.data["minimum"])
        else:
            values = [self.tree_get_value(st) for st in tree.get_branches()]
            value = None if None in values else max(values)
        self._tree_to_value[tree] = value
        return value

    def run(self):
        for tree in self.tree_graph.get_all_trees():
            value = self.tree_get_value(tree)
            if value is not None:
                tree.data["colour"] = self.value_to_color(value)


# ---------------------------------------------------------------------------
# Minimum & TransitionState data classes
# ---------------------------------------------------------------------------

class Minimum:
    """A minimum on the energy landscape.

    Parameters
    ----------
    energy : float
    coords : array-like
    _id : int
        Unique identifier used for hashing and equality.
    """

    def __init__(self, energy, coords, _id):
        self.energy = energy
        self.coords = np.asarray(coords, dtype=float)
        self._id = _id

    def id(self):
        return self._id

    def __eq__(self, other):
        if isinstance(other, Minimum):
            return self._id == other._id
        return self._id == other

    def __hash__(self):
        return self._id

    def __lt__(self, other):
        if isinstance(other, Minimum):
            return self._id < other._id
        return NotImplemented

    def __repr__(self):
        return f"Minimum(id={self._id}, energy={self.energy:.6g})"


class TransitionState:
    """A transition state (saddle point) connecting two minima.

    Parameters
    ----------
    energy : float
    coords : array-like
    min1, min2 : Minimum
    eigenval : float, optional
    eigenvec : array-like, optional
    """

    def __init__(self, energy, coords, min1, min2, eigenval=None, eigenvec=None):
        self.energy = energy
        self.coords = np.asarray(coords, dtype=float)
        self.minimum1 = min1
        self.minimum2 = min2
        self.eigenval = eigenval
        if eigenvec is not None:
            self.eigenvec = np.asarray(eigenvec, dtype=float)

    def __repr__(self):
        return (f"TransitionState(energy={self.energy:.6g}, "
                f"min1={self.minimum1._id}, min2={self.minimum2._id})")


# ---------------------------------------------------------------------------
# Bridge: hyperct Complex → SimpleGraph
# ---------------------------------------------------------------------------

def database_from_complex(HC, ts_energy="max"):
    """Build a ``SimpleGraph`` of ``Minimum`` / ``TransitionState`` objects
    from a ``hyperct.Complex`` that has a scalar field evaluated.

    Minimisers are identified using the hyperct vertex ``minimiser()`` method
    (i.e. ``v.f < nbr.f`` for every neighbour).  Each remaining vertex is
    assigned to the basin of its lowest-energy neighbour (processed in
    ascending energy order).  Transition states are placed at the
    lowest-energy boundary crossing between adjacent basins.

    Parameters
    ----------
    HC : Complex
        Must have been triangulated and had ``V.process_pools()`` called.
    ts_energy : str
        Strategy for estimating transition-state energy at basin boundaries:

        * ``"max"`` — ``max(f(v1), f(v2))`` at the boundary edge (default)
        * ``"midpoint"`` — ``0.5 * (f(v1) + f(v2))``

    Returns
    -------
    graph : SimpleGraph
        Graph with ``Minimum`` nodes and ``TransitionState`` edge attributes.
    energy_cache : dict
        Mapping from ``Minimum`` to energy, for use with
        ``DisconnectivityGraph(energy_cache=...)``.
    """
    if not hasattr(HC, 'V') or HC.V is None:
        raise ValueError("Complex must be triangulated first")

    # Collect vertices with finite scalar field values
    vertices = [v for v in HC.V
                if hasattr(v, 'f') and v.f is not None and np.isfinite(v.f)]
    if not vertices:
        raise ValueError("No vertices with finite field values")

    # Step 1: Identify minimisers using hyperct's own definition
    # (v.f < nbr.f for every neighbour in v.nn).
    minimisers = set(v for v in vertices if v.minimiser())
    if not minimisers:
        raise ValueError("No minimisers found in complex")

    # Step 2: Basin assignment — process vertices in ascending energy order.
    # Each minimiser seeds its own basin; every other vertex inherits the
    # basin of its lowest-energy already-assigned neighbour.
    basin_of: dict = {m: m for m in minimisers}

    for v in sorted(vertices, key=lambda v: float(v.f)):
        if v in basin_of:
            continue
        best = None
        for nbr in v.nn:
            if nbr in basin_of and (best is None or nbr.f < best.f):
                best = nbr
        if best is not None:
            basin_of[v] = basin_of[best]

    # Step 3: Create Minimum objects for minimiser vertices only
    graph = SimpleGraph()
    energy_cache = {}
    root_to_minimum: dict = {}

    roots = sorted(minimisers, key=lambda v: float(v.f))
    for _id, v in enumerate(roots):
        e = float(v.f)
        m = Minimum(energy=e, coords=v.x, _id=_id)
        root_to_minimum[v] = m
        graph.add_node(m)
        energy_cache[m] = e

    # Step 4: Find the lowest-energy boundary crossing between
    # each pair of adjacent basins.
    boundary: dict = {}  # (root_a, root_b) → (saddle_energy, midpoint)

    for v in vertices:
        rv = basin_of.get(v)
        if rv is None:
            continue
        for nbr in v.nn:
            rn = basin_of.get(nbr)
            if rn is None or rv is rn:
                continue
            if ts_energy == "midpoint":
                e = float(0.5 * (v.f + nbr.f))
            else:
                e = float(max(v.f, nbr.f))
            key = (rv, rn) if id(rv) < id(rn) else (rn, rv)
            if key not in boundary or e < boundary[key][0]:
                mid = 0.5 * (np.asarray(v.x, dtype=float)
                             + np.asarray(nbr.x, dtype=float))
                boundary[key] = (e, mid)

    # Step 5: Create TransitionState edges
    for (rv, rn), (e, coords) in boundary.items():
        m1 = root_to_minimum[rv]
        m2 = root_to_minimum[rn]
        ts = TransitionState(e, coords, m1, m2)
        graph.add_edge(m1, m2, ts=ts)

    return graph, energy_cache


# ---------------------------------------------------------------------------
# Main disconnectivity graph class
# ---------------------------------------------------------------------------

class DisconnectivityGraph:
    """Compute and plot a disconnectivity graph from an energy landscape.

    Parameters
    ----------
    graph : SimpleGraph
        Graph with ``Minimum`` nodes and ``TransitionState`` edge data
        (keyed as ``"ts"``).
    nlevels : int
        Number of energy levels for binning transition states.
    Emax : float, optional
        Maximum energy cutoff for transition states.
    minima : list of Minimum, optional
        Minima to ensure are displayed.
    subgraph_size : int, optional
        Include all connected components with at least this many nodes.
    order_by_energy : bool
        Order subtrees by minimum energy (low energy → centre).
    order_by_basin_size : bool
        Order subtrees by number of leaves (large basins → centre).
    center_gmin : bool
        Always place the global minimum's basin centrally.
    include_gmin : bool
        Include the global minimum even if not in the main cluster.
    node_offset : float
        Controls the angle of connecting lines (0 = horizontal, 1 = full angle).
    order_by_value : callable, optional
        ``order_by_value(minimum) → float`` for custom ordering.
    energy_cache : dict, optional
        Mapping ``node → energy`` overriding ``node.energy``.

    Examples
    --------
    >>> import hyperct
    >>> from hyperct._vis_disc import DisconnectivityGraph, database_from_complex
    >>> HC = hyperct.Complex(2, domain=[(-5, 5), (-5, 5)], sfield=lambda x: x[0]**2 + x[1]**2)
    >>> HC.triangulate()
    >>> HC.refine_all()
    >>> HC.V.process_pools()
    >>> graph, ecache = database_from_complex(HC)
    >>> dg = DisconnectivityGraph(graph, nlevels=10, energy_cache=ecache)
    >>> dg.calculate()
    >>> dg.plot()
    """

    def __init__(self, graph, minima=None, nlevels=20, Emax=None,
                 subgraph_size=None, order_by_energy=False,
                 order_by_basin_size=True, node_offset=1.,
                 center_gmin=True, include_gmin=True,
                 order_by_value=None, energy_cache=None):
        self.graph = graph
        self.nlevels = nlevels
        self.Emax = Emax
        self.subgraph_size = subgraph_size
        self.order_by_basin_size = order_by_basin_size
        self.order_by_energy = order_by_energy
        self.center_gmin = center_gmin
        self.gmin0 = None
        self.node_offset = node_offset
        self.get_value = order_by_value
        self.energy_cache = energy_cache if energy_cache is not None else {}
        if self.center_gmin:
            include_gmin = True

        if minima is None:
            minima = []
        self.min0list = list(minima)
        if include_gmin:
            elist = sorted((self._get_energy(m), m) for m in self.graph.nodes())
            if elist:
                self.gmin0 = elist[0][1]
                self.min0list.append(self.gmin0)

        self.transition_states = self.graph.get_edge_attributes("ts")

    # -- energy helpers -----------------------------------------------------

    def _get_energy(self, node):
        try:
            return self.energy_cache[node]
        except (KeyError, TypeError):
            return node.energy

    def set_energy_levels(self, elevels):
        self.elevels = elevels

    def _get_ts(self, min1, min2):
        try:
            return self.transition_states[(min1, min2)]
        except KeyError:
            return self.transition_states[(min2, min1)]

    # -- tree construction --------------------------------------------------

    def _make_tree(self, graph, energy_levels):
        transition_states = list(graph.get_edge_attributes("ts").values())
        minima = graph.nodes()
        maketree = _MakeTree(minima, transition_states, energy_levels,
                             get_energy=self._get_energy)
        trees = maketree.make_tree()
        self.minimum_to_leave = maketree.minimum_to_leave
        return trees

    # -- x-axis layout ------------------------------------------------------

    def _recursive_layout_x_axis(self, tree, xmin, dx_per_min):
        nminima = tree.number_of_leaves()
        subtrees = self._order_trees(tree.get_branches())
        tree.data["x"] = xmin + dx_per_min * nminima / 2.
        x = xmin
        for subtree in subtrees:
            self._recursive_layout_x_axis(subtree, x, dx_per_min)
            x += dx_per_min * subtree.number_of_leaves()

    def _layout_x_axis(self, tree):
        self._recursive_layout_x_axis(tree, 4.0, 1.)

    def _tree_get_minimum_energy(self, tree):
        return min(leaf.data["minimum"].energy for leaf in tree.get_leaves())

    # -- tree ordering ------------------------------------------------------

    def _order_trees(self, trees):
        if self.get_value is not None:
            return self._order_trees_by_value(trees)
        if self.order_by_energy:
            return self._order_trees_by_minimum_energy(trees)
        return self._order_trees_by_most_leaves(trees)

    def _order_trees_by_value(self, trees):
        def get_min_val(tree):
            return min(self.get_value(leaf.data["minimum"])
                       for leaf in tree.leaf_iterator())
        trees.sort(key=get_min_val)
        return trees

    def _order_trees_final(self, tree_value_list):
        mylist = sorted(tree_value_list, key=operator.itemgetter(0))
        neworder = deque()
        for i, item in enumerate(mylist):
            if i % 2 == 0:
                neworder.append(item[1])
            else:
                neworder.appendleft(item[1])
        return list(neworder)

    def _ensure_gmin_is_center(self, tree_value_list):
        if self.gmin0 is None:
            return tree_value_list
        for i, (v, tree) in enumerate(tree_value_list):
            if self.gmin0 in [leaf.data["minimum"] for leaf in tree.get_leaves()]:
                minvalue = min(val for val, _ in tree_value_list)
                tree_value_list[i] = (minvalue - 1, tree_value_list[i][1])
                break
        return tree_value_list

    def _order_trees_by_most_leaves(self, trees):
        mylist = [(tree.number_of_leaves(), tree) for tree in trees]
        if self.center_gmin:
            mylist = self._ensure_gmin_is_center(mylist)
        return self._order_trees_final(mylist)

    def _order_trees_by_minimum_energy(self, trees):
        mylist = [(self._tree_get_minimum_energy(tree), tree) for tree in trees]
        return self._order_trees_final(mylist)

    # -- line segment generation --------------------------------------------

    def _get_line_segment_single(self, line_segments, line_colours, tree, eoffset):
        color_default = (0., 0., 0.)
        if tree.parent is None:
            if "children_not_connected" in tree.data:
                treelist = tree.subtrees
            else:
                treelist = [tree]

            dy = self.energy_levels[-1] - self.energy_levels[-2]
            for t in treelist:
                x = t.data["x"]
                y = t.data["ethresh"]
                line_segments.append(([x, x], [y, y + dy]))
                line_colours.append(color_default)
        else:
            xparent = tree.parent.data['x']
            xself = tree.data['x']
            yparent = tree.parent.data["ethresh"]

            if tree.is_leaf():
                yself = self._get_energy(tree.data["minimum"])
            else:
                yself = tree.data["ethresh"]

            yhigh = yparent - eoffset
            draw_vertical = yhigh > yself
            if not draw_vertical:
                yhigh = yself

            color = tree.data.get('colour', color_default)

            if tree.is_leaf() and not draw_vertical:
                if "_x_updated" not in tree.data:
                    dxdy = (xself - xparent) / eoffset
                    xself = dxdy * (yparent - yself) + xparent
                    tree.data['x'] = xself
                    tree.data["_x_updated"] = True
            else:
                line_segments.append(([xself, xself], [yself, yhigh]))
                line_colours.append(color)

            if "children_not_connected" not in tree.parent.data:
                line_segments.append(([xself, xparent], [yhigh, yparent]))
                line_colours.append(color)

    def _get_line_segment_recursive(self, line_segments, line_colours, tree, eoffset):
        self._get_line_segment_single(line_segments, line_colours, tree, eoffset)
        for subtree in tree.get_branches():
            self._get_line_segment_recursive(line_segments, line_colours, subtree, eoffset)

    def _get_line_segments(self, tree, eoffset=-1.):
        line_segments = []
        line_colours = []
        self._get_line_segment_recursive(line_segments, line_colours, tree, eoffset)
        return line_segments, line_colours

    # -- graph reduction ----------------------------------------------------

    def _remove_nodes_with_few_edges(self, graph, nmin):
        rmlist = [n for n in graph.nodes() if graph.degree(n) < nmin]
        for n in rmlist:
            graph.remove_node(n)
        return graph

    def _remove_high_energy_minima(self, graph, emax):
        if emax is None:
            return graph
        rmlist = [m for m in graph.nodes() if self._get_energy(m) > emax]
        for m in rmlist:
            graph.remove_node(m)
        return graph

    def _remove_high_energy_transitions(self, graph, emax):
        if emax is None:
            return graph
        rmlist = [edge for edge in graph.edges()
                  if self._get_energy(self._get_ts(edge[0], edge[1])) > emax]
        for u, v in rmlist:
            graph.remove_edge(u, v)
        return graph

    def _reduce_graph(self, graph, min0list):
        used_nodes = set()
        for min0 in min0list:
            try:
                comp = graph.node_connected_component(min0)
            except KeyError:
                continue
            if len(comp) > 2:
                used_nodes |= comp

        if not used_nodes:
            components = sorted(graph.connected_components(), key=len, reverse=True)
            if components:
                used_nodes = components[0]

        if self.subgraph_size is not None:
            for comp in graph.connected_components():
                if len(comp) >= self.subgraph_size:
                    used_nodes |= comp

        return graph.subgraph(used_nodes)

    # -- public API ---------------------------------------------------------

    def get_minima_layout(self):
        leaves = self.tree_graph.get_leaves()
        minima = [leaf.data["minimum"] for leaf in leaves]
        xpos = [leaf.data["x"] for leaf in leaves]
        return xpos, minima

    def _get_energy_levels(self, graph):
        if hasattr(self, "elevels"):
            return self.elevels
        elist = [self._get_energy(self._get_ts(*edge)) for edge in graph.edges()]
        if not elist:
            raise ValueError("graph has no edges — is the global minimum connected?")
        emin = min(elist)
        emax = self.Emax if self.Emax is not None else max(elist)
        de = (emax - emin) / (self.nlevels - 1)
        if de == 0:
            # All transition states have the same energy; expand the range
            # so that at least one level sits strictly above the TS energies.
            de = max(abs(emax), 1.0) * 0.1
        return [emin + de * i for i in range(self.nlevels)]

    def calculate(self):
        """Compute the disconnectivity tree.  Must be called before ``plot()``."""
        graph = self.graph.copy()
        assert graph.number_of_nodes() > 0, "graph has no minima"
        assert graph.number_of_edges() > 0, "graph has no transition states"

        graph = self._remove_high_energy_minima(graph, self.Emax)
        graph = self._remove_high_energy_transitions(graph, self.Emax)
        assert graph.number_of_nodes() > 0, "after Emax cutoff, graph has no minima"
        assert graph.number_of_edges() > 0, "after Emax cutoff, graph has no edges"

        graph = self._reduce_graph(graph, self.min0list)

        elevels = self._get_energy_levels(graph)
        self.energy_levels = elevels

        graph = self._remove_high_energy_minima(graph, elevels[-1])
        graph = self._remove_high_energy_transitions(graph, elevels[-1])
        graph = self._remove_nodes_with_few_edges(graph, 1)

        assert graph.number_of_nodes() > 0, "after cleanup, graph has no minima"
        assert graph.number_of_edges() > 0, "after cleanup, graph has no edges"

        tree_graph = self._make_tree(graph, elevels)
        self._layout_x_axis(tree_graph)

        eoffset = (elevels[-1] - elevels[-2]) * self.node_offset
        self.eoffset = eoffset
        self.tree_graph = tree_graph

    # -- colouring convenience methods --------------------------------------

    def color_by_group(self, groups, colors=None):
        """Colour nodes by grouping of minima."""
        ColorDGraphByGroups(self.tree_graph, groups, colors=colors).run()

    def color_by_value(self, minimum_to_value, colormap=None, normalize_values=True):
        """Colour nodes by a scalar value per minimum."""
        ColorDGraphByValue(self.tree_graph, minimum_to_value,
                           colormap=colormap,
                           normalize_values=normalize_values).run()

    # -- plotting -----------------------------------------------------------

    def draw_minima(self, minima, axes=None, **kwargs):
        """Draw specified minima as scatter points on the graph."""
        if axes is None:
            axes = self.axes
        kwargs.setdefault("marker", "o")
        xpos, minlist = self.get_minima_layout()
        m2x = dict(zip(minlist, xpos))
        minima = list(minima)
        xpos = [m2x[m] for m in minima]
        energies = [m.energy for m in minima]
        axes.scatter(xpos, energies, **kwargs)

    def plot(self, show_minima=False, linewidth=0.5, axes=None, title=None):
        """Draw the disconnectivity graph with matplotlib.

        Call ``calculate()`` first.
        """
        from matplotlib.collections import LineCollection
        import matplotlib.pyplot as plt

        self.line_segments, self.line_colours = self._get_line_segments(
            self.tree_graph, eoffset=self.eoffset)

        if axes is not None:
            ax = axes
        elif hasattr(self, 'axes'):
            ax = self.axes
        else:
            fig = plt.figure(figsize=(6, 7))
            fig.set_facecolor('white')
            ax = fig.add_subplot(111, adjustable='box')

        ax.tick_params(axis='y', direction='out')
        ax.yaxis.tick_left()
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')

        if title is not None:
            ax.set_title(title)

        if show_minima:
            xpos, minima = self.get_minima_layout()
            energies = [m.energy for m in minima]
            ax.plot(xpos, energies, 'o')

        linecollection = LineCollection(
            [[(x[0], y[0]), (x[1], y[1])] for x, y in self.line_segments])
        linecollection.set_linewidth(linewidth)
        linecollection.set_color(self.line_colours)
        ax.add_collection(linecollection)

        ax.autoscale_view(scalex=True, scaley=True, tight=False)
        ax.set_ybound(upper=self.Emax)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_xticks([])
        self.axes = ax

    def label_minima(self, minima_labels, axes=None, rotation=60., **kwargs):
        """Label specified minima on the x-axis.

        Parameters
        ----------
        minima_labels : dict
            ``{minimum: label_string}``
        """
        ax = axes if axes is not None else self.axes
        leaves = [leaf for leaf in self.tree_graph.leaf_iterator()
                  if leaf.data["minimum"] in minima_labels]
        xpos = [leaf.data["x"] for leaf in leaves]
        labels = [minima_labels[leaf.data["minimum"]] for leaf in leaves]
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=rotation, **kwargs)

    def show(self):  # pragma: no cover
        import matplotlib.pyplot as plt
        plt.show()

    def savefig(self, *args, **kwargs):  # pragma: no cover
        import matplotlib.pyplot as plt
        plt.savefig(*args, **kwargs)

    # -- class method for hyperct integration --------------------------------

    @classmethod
    def from_complex(cls, HC, nlevels=20, ts_energy="max", **kwargs):
        """Build a ``DisconnectivityGraph`` directly from a hyperct ``Complex``.

        Parameters
        ----------
        HC : Complex
            Must have a scalar field and ``V.process_pools()`` called.
        nlevels : int
            Number of energy levels.
        ts_energy : str
            ``"max"`` or ``"midpoint"`` — how to estimate TS energy.
        **kwargs
            Additional keyword arguments passed to ``DisconnectivityGraph``.

        Returns
        -------
        DisconnectivityGraph
        """
        graph, energy_cache = database_from_complex(HC, ts_energy=ts_energy)
        return cls(graph, nlevels=nlevels, energy_cache=energy_cache, **kwargs)
