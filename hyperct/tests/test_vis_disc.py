"""Tests for disconnectivity graph visualization module."""
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import for headless rendering

import pytest
import numpy as np
from matplotlib import pyplot

from hyperct._vis_disc import (
    SimpleGraph, UnionFind, Tree, DGTree, TreeLeastCommonAncestor,
    Minimum, TransitionState, DisconnectivityGraph, database_from_complex
)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    pyplot.close('all')


# ---------------------------------------------------------------------------
# TestSimpleGraph
# ---------------------------------------------------------------------------

class TestSimpleGraph:
    """Tests for SimpleGraph lightweight undirected graph."""

    def test_add_node_and_nodes_from(self):
        """Add nodes individually and in bulk."""
        g = SimpleGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_nodes_from(["c", "d", "e"])
        assert g.number_of_nodes() == 5
        assert set(g.nodes()) == {"a", "b", "c", "d", "e"}

    def test_remove_node(self):
        """Remove a node and its adjacent edges."""
        g = SimpleGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.remove_node("b")
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 0
        assert "b" not in g.nodes()

    def test_add_edge_and_edges_from(self):
        """Add edges individually and in bulk."""
        g = SimpleGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edges_from([("b", "c"), ("c", "d", {"weight": 2.0})])
        assert g.number_of_edges() == 3
        assert g.get_edge_data("a", "b")["weight"] == 1.0
        assert g.get_edge_data("c", "d")["weight"] == 2.0

    def test_remove_edge(self):
        """Remove an edge from the graph."""
        g = SimpleGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.remove_edge("a", "b")
        assert g.number_of_edges() == 1
        assert g.get_edge_data("a", "b") is None

    def test_degree(self):
        """Check node degree calculation."""
        g = SimpleGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("a", "d")
        assert g.degree("a") == 3
        assert g.degree("b") == 1

    def test_get_edge_attributes(self):
        """Retrieve edge attributes by name."""
        g = SimpleGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=2.0)
        weights = g.get_edge_attributes("weight")
        assert len(weights) == 2
        assert 1.0 in weights.values()
        assert 2.0 in weights.values()

    def test_connected_components(self):
        """Find all connected components."""
        g = SimpleGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("d", "e")
        components = g.connected_components()
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [2, 3]

    def test_node_connected_component(self):
        """Find the component containing a specific node."""
        g = SimpleGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("d", "e")
        comp = g.node_connected_component("b")
        assert comp == {"a", "b", "c"}

    def test_subgraph(self):
        """Create a subgraph with specified nodes."""
        g = SimpleGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=2.0)
        g.add_edge("c", "d", weight=3.0)
        sg = g.subgraph(["a", "b", "c"])
        assert sg.number_of_nodes() == 3
        assert sg.number_of_edges() == 2
        assert "d" not in sg.nodes()

    def test_copy(self):
        """Copy a graph with all nodes and edges."""
        g = SimpleGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=2.0)
        g2 = g.copy()
        assert g2.number_of_nodes() == 3
        assert g2.number_of_edges() == 2
        assert g2.get_edge_data("a", "b")["weight"] == 1.0
        # Verify it's a copy, not a reference
        g2.add_edge("c", "d")
        assert g.number_of_edges() == 2
        assert g2.number_of_edges() == 3


# ---------------------------------------------------------------------------
# TestUnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    """Tests for UnionFind weighted quick-union with path compression."""

    def test_singleton_creation(self):
        """Create singleton sets via __getitem__."""
        uf = UnionFind()
        root_a = uf["a"]
        root_b = uf["b"]
        assert root_a == "a"
        assert root_b == "b"
        assert root_a != root_b

    def test_union_two_elements(self):
        """Union two elements into the same set."""
        uf = UnionFind()
        uf["a"]
        uf["b"]
        uf.union("a", "b")
        assert uf["a"] == uf["b"]

    def test_path_compression(self):
        """Path compression makes all nodes point to root."""
        uf = UnionFind()
        uf["a"]
        uf["b"]
        uf["c"]
        uf.union("a", "b")
        uf.union("b", "c")
        # All three should have the same root after path compression
        root = uf["a"]
        assert uf["b"] == root
        assert uf["c"] == root

    def test_groups_iter(self):
        """Iterate over current root elements."""
        uf = UnionFind()
        uf["a"]
        uf["b"]
        uf["c"]
        uf.union("a", "b")
        roots = list(uf.groups_iter())
        assert len(roots) == 2  # Two separate groups: {a,b} and {c}

    def test_transitive_union(self):
        """Transitive union: a-b, b-c => a, b, c same group."""
        uf = UnionFind()
        uf["a"]
        uf["b"]
        uf["c"]
        uf.union("a", "b")
        uf.union("b", "c")
        root = uf["a"]
        assert uf["b"] == root
        assert uf["c"] == root


# ---------------------------------------------------------------------------
# TestTree
# ---------------------------------------------------------------------------

class TestTree:
    """Tests for Tree generic tree node."""

    def test_is_leaf(self):
        """A tree with no children is a leaf."""
        root = Tree()
        assert root.is_leaf()
        child = Tree(parent=root)
        assert child.is_leaf()
        assert not root.is_leaf()

    def test_add_branch_and_make_branch(self):
        """Add and create branches."""
        root = Tree()
        child1 = Tree()
        root.add_branch(child1)
        assert child1.parent == root
        assert len(root.subtrees) == 1

        child2 = root.make_branch()
        assert child2.parent == root
        assert len(root.subtrees) == 2

    def test_number_of_leaves(self):
        """Count leaves recursively."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        grandchild = Tree(parent=child1)
        # child2 and grandchild are leaves
        assert root.number_of_leaves() == 2

    def test_get_leaves(self):
        """Retrieve all leaf nodes."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        grandchild = Tree(parent=child1)
        leaves = root.get_leaves()
        # child2 and grandchild are leaves
        assert len(leaves) == 2
        assert grandchild in leaves
        assert child2 in leaves

    def test_leaf_iterator(self):
        """Iterate over leaves."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        Tree(parent=child1)
        leaves = list(root.leaf_iterator())
        # child2 and the grandchild are leaves
        assert len(leaves) == 2

    def test_get_ancestors(self):
        """Get ancestors of a node."""
        root = Tree()
        child = Tree(parent=root)
        grandchild = Tree(parent=child)
        ancestors = list(grandchild.get_ancestors())
        assert ancestors == [child, root]

    def test_get_all_trees(self):
        """Get all nodes in the tree."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        grandchild = Tree(parent=child1)
        all_trees = list(root.get_all_trees())
        assert len(all_trees) == 4
        assert root in all_trees
        assert grandchild in all_trees

    def test_number_of_subtrees(self):
        """Count all subtrees including self."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        Tree(parent=child1)
        assert root.number_of_subtrees() == 4


# ---------------------------------------------------------------------------
# TestTreeLeastCommonAncestor
# ---------------------------------------------------------------------------

class TestTreeLeastCommonAncestor:
    """Tests for TreeLeastCommonAncestor."""

    def test_lca_of_siblings(self):
        """LCA of siblings is their parent."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        lca = TreeLeastCommonAncestor([child1, child2])
        assert lca.least_common_ancestor == root

    def test_lca_different_depths(self):
        """LCA of nodes at different depths."""
        root = Tree()
        child1 = Tree(parent=root)
        child2 = Tree(parent=root)
        grandchild = Tree(parent=child1)
        lca = TreeLeastCommonAncestor([grandchild, child2])
        assert lca.least_common_ancestor == root

    def test_lca_no_common_ancestor_raises(self):
        """Error when trees have no common ancestor."""
        root1 = Tree()
        root2 = Tree()
        child1 = Tree(parent=root1)
        child2 = Tree(parent=root2)
        with pytest.raises((ValueError, AssertionError)):
            TreeLeastCommonAncestor([child1, child2])


# ---------------------------------------------------------------------------
# TestDGTree
# ---------------------------------------------------------------------------

class TestDGTree:
    """Tests for DGTree specialized tree for disconnectivity graphs."""

    def test_contains_minimum(self):
        """Check if tree contains a specific minimum."""
        root = DGTree()
        leaf1 = DGTree(parent=root)
        leaf2 = DGTree(parent=root)
        m1 = Minimum(1.0, [0.0], 1)
        m2 = Minimum(2.0, [1.0], 2)
        leaf1.data["minimum"] = m1
        leaf2.data["minimum"] = m2
        assert root.contains_minimum(m1)
        assert root.contains_minimum(m2)
        assert not root.contains_minimum(Minimum(3.0, [2.0], 3))

    def test_get_minima(self):
        """Get all minima in the tree."""
        root = DGTree()
        leaf1 = DGTree(parent=root)
        leaf2 = DGTree(parent=root)
        m1 = Minimum(1.0, [0.0], 1)
        m2 = Minimum(2.0, [1.0], 2)
        leaf1.data["minimum"] = m1
        leaf2.data["minimum"] = m2
        minima = root.get_minima()
        assert len(minima) == 2
        assert m1 in minima
        assert m2 in minima

    def test_get_one_minimum(self):
        """Get one minimum from the tree (cached)."""
        root = DGTree()
        branch = DGTree(parent=root)
        leaf = DGTree(parent=branch)
        m = Minimum(1.0, [0.0], 1)
        leaf.data["minimum"] = m
        min_found = root.get_one_minimum()
        assert min_found == m
        # Second call should use cache
        min_found2 = root.get_one_minimum()
        assert min_found2 == m


# ---------------------------------------------------------------------------
# TestMinimumTransitionState
# ---------------------------------------------------------------------------

class TestMinimumTransitionState:
    """Tests for Minimum and TransitionState data classes."""

    def test_minimum_equality_by_id(self):
        """Minima are equal by _id."""
        m1 = Minimum(1.0, [0.0, 0.0], 1)
        m2 = Minimum(2.0, [1.0, 1.0], 1)
        m3 = Minimum(1.0, [0.0, 0.0], 2)
        assert m1 == m2
        assert m1 != m3

    def test_minimum_hash(self):
        """Minimum hash is based on _id."""
        m1 = Minimum(1.0, [0.0, 0.0], 1)
        m2 = Minimum(1.0, [0.0, 0.0], 1)
        assert hash(m1) == hash(m2)
        # Can use as dict key
        d = {m1: "value"}
        assert d[m2] == "value"

    def test_transition_state_creation(self):
        """Create TransitionState with eigenvalue/eigenvec."""
        m1 = Minimum(1.0, [0.0], 1)
        m2 = Minimum(2.0, [1.0], 2)
        ts = TransitionState(1.5, [0.5], m1, m2, eigenval=-1.0, eigenvec=[1.0])
        assert ts.energy == 1.5
        assert np.allclose(ts.coords, [0.5])
        assert ts.minimum1 == m1
        assert ts.minimum2 == m2
        assert ts.eigenval == -1.0

    def test_minimum_repr(self):
        """Check Minimum repr string."""
        m = Minimum(1.23456, [0.0], 42)
        r = repr(m)
        assert "Minimum" in r
        assert "id=42" in r
        assert "1.23456" in r

    def test_transition_state_repr(self):
        """Check TransitionState repr string."""
        m1 = Minimum(1.0, [0.0], 1)
        m2 = Minimum(2.0, [1.0], 2)
        ts = TransitionState(1.5, [0.5], m1, m2)
        r = repr(ts)
        assert "TransitionState" in r
        assert "energy=1.5" in r
        assert "min1=1" in r
        assert "min2=2" in r


# ---------------------------------------------------------------------------
# TestDisconnectivityGraph
# ---------------------------------------------------------------------------

class TestDisconnectivityGraph:
    """Tests for DisconnectivityGraph computation and plotting."""

    def test_build_graph_manually(self):
        """Build graph manually with Minimum/TransitionState, calculate."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(1.0, [1.0], 2)
        m3 = Minimum(0.5, [0.5], 3)
        graph.add_node(m1)
        graph.add_node(m2)
        graph.add_node(m3)
        ts12 = TransitionState(2.0, [0.5], m1, m2)
        ts23 = TransitionState(1.5, [0.75], m2, m3)
        graph.add_edge(m1, m2, ts=ts12)
        graph.add_edge(m2, m3, ts=ts23)

        dg = DisconnectivityGraph(graph, nlevels=5)
        dg.calculate()
        assert hasattr(dg, 'tree_graph')
        assert dg.tree_graph is not None

    def test_plot_produces_line_segments(self):
        """plot() produces line_segments."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(1.0, [1.0], 2)
        m3 = Minimum(0.5, [0.5], 3)
        graph.add_node(m1)
        graph.add_node(m2)
        graph.add_node(m3)
        ts12 = TransitionState(2.0, [0.5], m1, m2)
        ts23 = TransitionState(1.5, [0.75], m2, m3)
        graph.add_edge(m1, m2, ts=ts12)
        graph.add_edge(m2, m3, ts=ts23)

        dg = DisconnectivityGraph(graph, nlevels=5)
        dg.calculate()
        dg.plot()
        assert hasattr(dg, 'line_segments')
        assert len(dg.line_segments) > 0

    def test_get_minima_layout(self):
        """get_minima_layout returns correct positions."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(1.0, [1.0], 2)
        m3 = Minimum(0.5, [0.5], 3)
        graph.add_node(m1)
        graph.add_node(m2)
        graph.add_node(m3)
        ts12 = TransitionState(1.5, [0.5], m1, m2)
        ts23 = TransitionState(1.8, [0.75], m2, m3)
        ts13 = TransitionState(1.6, [0.25], m1, m3)
        graph.add_edge(m1, m2, ts=ts12)
        graph.add_edge(m2, m3, ts=ts23)
        graph.add_edge(m1, m3, ts=ts13)

        dg = DisconnectivityGraph(graph, nlevels=3, center_gmin=False, include_gmin=False)
        dg.calculate()
        xpos, minima = dg.get_minima_layout()
        assert len(xpos) == len(minima)
        assert len(minima) >= 2  # At least some minima present

    def test_order_by_energy(self):
        """order_by_energy option orders subtrees."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(2.0, [1.0], 2)
        m3 = Minimum(1.0, [0.5], 3)
        graph.add_node(m1)
        graph.add_node(m2)
        graph.add_node(m3)
        ts12 = TransitionState(3.0, [0.5], m1, m2)
        ts13 = TransitionState(2.5, [0.25], m1, m3)
        graph.add_edge(m1, m2, ts=ts12)
        graph.add_edge(m1, m3, ts=ts13)

        dg = DisconnectivityGraph(graph, nlevels=5, order_by_energy=True)
        dg.calculate()
        assert hasattr(dg, 'tree_graph')

    def test_set_energy_levels_manual_override(self):
        """set_energy_levels manual override."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(1.0, [1.0], 2)
        graph.add_node(m1)
        graph.add_node(m2)
        ts = TransitionState(2.0, [0.5], m1, m2)
        graph.add_edge(m1, m2, ts=ts)

        dg = DisconnectivityGraph(graph, nlevels=3)
        custom_levels = [0.0, 1.0, 2.0, 3.0]
        dg.set_energy_levels(custom_levels)
        dg.calculate()
        assert dg.energy_levels == custom_levels

    def test_emax_cutoff(self):
        """Emax cutoff removes high-energy nodes."""
        graph = SimpleGraph()
        m1 = Minimum(0.0, [0.0], 1)
        m2 = Minimum(1.0, [1.0], 2)
        m3 = Minimum(10.0, [2.0], 3)
        graph.add_node(m1)
        graph.add_node(m2)
        graph.add_node(m3)
        ts12 = TransitionState(2.0, [0.5], m1, m2)
        ts23 = TransitionState(12.0, [1.5], m2, m3)
        graph.add_edge(m1, m2, ts=ts12)
        graph.add_edge(m2, m3, ts=ts23)

        dg = DisconnectivityGraph(graph, nlevels=5, Emax=5.0)
        dg.calculate()
        # m3 should be excluded due to high energy
        xpos, minima = dg.get_minima_layout()
        assert m3 not in minima


# ---------------------------------------------------------------------------
# TestFromComplex
# ---------------------------------------------------------------------------

class TestFromComplex:
    """Tests for database_from_complex and DisconnectivityGraph.from_complex."""

    @staticmethod
    def _double_well(x):
        """Two basins at x0=±1: (x0²-1)² + x1²."""
        return (x[0]**2 - 1)**2 + x[1]**2

    def test_database_from_complex_builds_graph(self):
        """database_from_complex creates only local minima as nodes."""
        import hyperct
        HC = hyperct.Complex(2, domain=[(-2, 2), (-2, 2)],
                             sfield=self._double_well)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        HC.refine_all()
        HC.V.process_pools()

        graph, ecache = database_from_complex(HC)
        nodes = graph.nodes()
        # Double-well should yield 2 basins and at least 1 edge
        assert graph.number_of_nodes() >= 2
        assert graph.number_of_edges() >= 1
        assert len(ecache) == graph.number_of_nodes()
        assert all(isinstance(n, Minimum) for n in nodes)

    def test_from_complex_builds_and_calculates(self):
        """DisconnectivityGraph.from_complex builds and calculates."""
        import hyperct
        HC = hyperct.Complex(2, domain=[(-2, 2), (-2, 2)],
                             sfield=self._double_well)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        HC.refine_all()
        HC.V.process_pools()

        dg = DisconnectivityGraph.from_complex(HC, nlevels=5)
        dg.calculate()
        assert hasattr(dg, 'tree_graph')
        assert dg.tree_graph is not None
        # Should find the two basins
        xpos, minima = dg.get_minima_layout()
        assert len(minima) >= 2

    def test_from_complex_plot_runs(self):
        """from_complex plot runs without error."""
        import hyperct

        def multi_basin(x):
            return np.sin(3 * x[0]) + x[0]**2 + x[1]**2

        HC = hyperct.Complex(2, domain=[(-3, 3), (-3, 3)],
                             sfield=multi_basin)
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        HC.refine_all()
        HC.V.process_pools()

        dg = DisconnectivityGraph.from_complex(HC, nlevels=10)
        dg.calculate()
        dg.plot()
        assert hasattr(dg, 'line_segments')
        assert len(dg.line_segments) > 0
