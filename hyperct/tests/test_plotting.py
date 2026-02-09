"""Tests for Complex plotting methods."""
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import for headless rendering

import os
import pytest
import numpy
from unittest.mock import patch
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch

from hyperct._complex import Complex


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    pyplot.close('all')


def simple_sfield(x):
    """Simple scalar field for testing: sum of squares."""
    return numpy.sum(numpy.array(x) ** 2)


# --- Tests for plot_complex orchestrator ---

class TestPlotComplex:
    """Tests for the main Complex.plot_complex() method."""

    def test_1d_no_field(self):
        """1D complex without field renders and returns fig/ax tuple."""
        H = Complex(1, domain=[(0.0, 5.0)])
        H.triangulate()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(show=False, save_fig=False)
        assert fig_c is not None
        assert ax_c is not None

    def test_1d_with_field(self):
        """1D complex with scalar field renders surface figure."""
        H = Complex(1, sfield=lambda x: float(x) ** 2, domain=[(0.0, 5.0)])
        H.triangulate()
        H.V.process_pools()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(show=False, save_fig=False)
        assert fig_c is not None
        assert fig_s is not None

    def test_2d_no_field(self):
        """2D complex without field, contour/surface disabled."""
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(
            show=False, save_fig=False,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert ax_c is not None

    def test_2d_with_field_and_contour(self):
        """2D complex with field, contour and surface enabled."""
        H = Complex(2, sfield=simple_sfield,
                    domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        H.V.process_pools()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(show=False, save_fig=False)
        assert fig_s is not None
        assert ax_s is not None

    def test_3d_complex(self):
        """3D complex renders scatter and edges."""
        H = Complex(3, domain=[(0.0, 5.0)] * 3)
        H.triangulate()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(show=False, save_fig=False)
        assert ax_c is not None

    def test_4d_returns_none(self):
        """Dimensions > 3 should return None figures with a warning."""
        H = Complex(4)
        H.triangulate()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(show=False, save_fig=False)
        assert fig_c is None
        assert ax_c is None

    def test_no_grids_option(self):
        """no_grids=True removes axis ticks and frame."""
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        H.plot_complex(
            show=False, save_fig=False, no_grids=True,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert len(H.ax_complex.get_xticks()) == 0

    def test_custom_fig_ax_passthrough(self):
        """Passing external fig/ax should use them, not create new ones."""
        fig, ax = pyplot.subplots()
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        H.plot_complex(
            show=False, save_fig=False,
            fig_complex=fig, ax_complex=ax,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert H.fig_complex is fig
        assert H.ax_complex is ax

    def test_save_fig_creates_file(self, tmp_path):
        """save_fig=True writes a file to the specified path."""
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        strpath = str(tmp_path / "test_output.pdf")
        H.plot_complex(
            show=False, save_fig=True, strpath=strpath,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert os.path.exists(strpath)

    def test_2d_with_splits(self):
        """2D complex with refinement still plots correctly."""
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        H.refine_all()
        fig_c, ax_c, fig_s, ax_s = H.plot_complex(
            show=False, save_fig=False,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert ax_c is not None


# --- Tests for plot helper methods ---

class TestPlotClean:
    """Tests for plot_clean method."""

    def test_clean_removes_attributes(self):
        """plot_clean() should delete fig/ax attributes."""
        H = Complex(2, domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        H.plot_complex(
            show=False, save_fig=False,
            contour_plot=False, surface_plot=False, surface_field_plot=False
        )
        assert hasattr(H, 'ax_complex')
        assert hasattr(H, 'fig_complex')
        H.plot_clean()
        assert not hasattr(H, 'ax_complex')
        assert not hasattr(H, 'fig_complex')

    def test_clean_idempotent(self):
        """Calling plot_clean twice should not raise."""
        H = Complex(2)
        H.triangulate()
        H.plot_clean()
        H.plot_clean()  # Should not raise


class TestPlotSubMethods:
    """Tests for individual plot helper methods."""

    def test_plot_directed_edge_2d(self):
        """2D directed edge returns a FancyArrowPatch."""
        H = Complex(2)
        H.triangulate()
        ap = H.plot_directed_edge(
            10.0, 5.0, [0.0, 0.0], [1.0, 1.0], proj_dim=2
        )
        assert isinstance(ap, FancyArrowPatch)

    def test_plot_directed_edge_3d(self):
        """3D directed edge returns an Arrow3D."""
        from hyperct._misc import Arrow3D
        H = Complex(3)
        H.triangulate()
        ap = H.plot_directed_edge(
            10.0, 5.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], proj_dim=3
        )
        assert isinstance(ap, Arrow3D)

    def test_plot_min_points_2d(self):
        """plot_min_points returns the axes object."""
        fig, ax = pyplot.subplots()
        H = Complex(2)
        H.triangulate()
        result = H.plot_min_points(ax, [[0.5, 0.5]], proj_dim=2)
        assert result is ax

    def test_plot_min_points_3d(self):
        """plot_min_points works in 3D."""
        fig = pyplot.figure()
        ax = fig.add_subplot(projection='3d')
        H = Complex(3)
        H.triangulate()
        result = H.plot_min_points(ax, [[0.5, 0.5, 0.5]], proj_dim=3)
        assert result is ax

    def test_plot_field_grids_caching(self):
        """Second call to plot_field_grids should return cached arrays."""
        H = Complex(2, sfield=simple_sfield,
                    domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        xg1, yg1, Z1 = H.plot_field_grids(H.bounds, simple_sfield, ())
        xg2, yg2, Z2 = H.plot_field_grids(H.bounds, simple_sfield, ())
        assert xg1 is xg2  # Same object (cached)
        assert Z1 is Z2

    def test_plot_field_grids_shape(self):
        """plot_field_grids returns arrays of correct shape."""
        H = Complex(2, sfield=simple_sfield,
                    domain=[(0.0, 5.0), (0.0, 5.0)])
        H.triangulate()
        xg, yg, Z = H.plot_field_grids(H.bounds, simple_sfield, ())
        assert xg.shape == yg.shape
        assert xg.shape == Z.shape
