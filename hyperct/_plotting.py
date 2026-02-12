"""
Standalone plotting functions for the Complex class.

This module provides module-level functions for visualizing simplicial complexes.
Each function that operates on a Complex instance takes it as the first argument `hc`.
"""
import os
import logging
import decimal

import numpy

# Optional matplotlib import
try:
    import matplotlib
    from matplotlib import pyplot
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d import axes3d, Axes3D, proj3d
    from ._misc import Arrow3D
except ImportError:
    logging.warning("Plotting functions are unavailable. To use install "
                    "matplotlib, install using ex. `pip install matplotlib` ")
    matplotlib_available = False
else:
    matplotlib_available = True


def plot_complex(hc, show=True, directed=True, complex_plot=True,
                contour_plot=True, surface_plot=True,
                surface_field_plot=True, minimiser_points=True,
                point_color='do', line_color='do',
                complex_color_f='lo', complex_color_e='do', pointsize=7,
                no_grids=False, save_fig=True, strpath=None,
                plot_path='fig/', fig_name='complex.pdf', arrow_width=None,
                fig_surface=None, ax_surface=None, fig_complex=None,
                ax_complex=None
                ):
    """
    Plots the current simplicial complex contained in the class. It requires
    at least one vector in the hc.V to have been defined.


    :param hc: Complex instance
    :param show: boolean, optional, show the output plots
    :param directed: boolean, optional, adds directed arrows to edges
    :param contour_plot: boolean, optional, contour plots of the field functions
    :param surface_plot: boolean, optional, a 3 simplicial complex + sfield plot
    :param surface_field_plot: boolean, optional, 3 dimensional surface + contour plot
    :param minimiser_points: boolean, optional, adds minimiser points
    :param point_color: str or vec, optional, colour of complex points
    :param line_color: str or vec, optional, colour of complex edges
    :param complex_color_f: str or vec, optional, colour of surface complex faces
    :param complex_color_e: str or vec, optional, colour of surface complex edges
    :param pointsize: float, optional, size of vectices on plots
    :param no_grids: boolean, optional, removes gridlines and axes
    :param save_fig: boolean, optional, save the output figure to file
    :param strpath: str, optional, string path of the file name
    :param plot_path: str, optional, relative path to file outputs
    :param fig_name: str, optional, name of the complex file to save
    :param arrow_width: float, optional, fixed size for arrows
    :return: hc.ax_complex, a matplotlib Axes class containing the complex and field contour
    :return: hc.ax_surface, a matplotlib Axes class containing the complex surface and field surface
    TODO: hc.fig_* missing
    Examples
    --------
    # Initiate a complex class
    >>> import pylab
    >>> H = Complex(2, domain=[(0, 10)], sfield=func)

    # As an example we'll use the built in triangulation to generate vertices
    >>> H.triangulate()
    >>> H.split_generation()

    # Plot the complex
    >>> plot_complex(H)

    # You can add any sort of custom drawings to the Axes classes of the
    plots
    >>> H.ax_complex.plot(0.25, 0.25, '.', color='k', markersize=10)
    >>> H.ax_surface.scatter(0.25, 0.25, 0.25, '.', color='k', s=10)

    # Show the final plot
    >>> pylab.show()

    # Clear current plot instances
    >>> plot_clean(H)

    Example 2: Subplots  #TODO: Test
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(ncols=2)
    >>> H = Complex(2, domain=[(0, 10)], sfield=func)
    >>> H.triangulate()
    >>> H.split_generation()

    # Plot the complex on the same subplot
    >>> plot_complex(H, fig_surface=fig, ax_surface=axes[0],
    ...              fig_complex=fig, ax_complex=axes[1])

    # Note you can also plot several complex objects on larger subplots
    #  using this method.

    """
    if not matplotlib_available:
        logging.warning("Plotting functions are unavailable. To "
                        "install matplotlib install using ex. `pip install "
                        "matplotlib` ")
        return
    if hc.sfield is None:
        directed = False  #TODO: We used this to avoid v.minimiser_point
                          # errors when is no field, should check for field
                          # instead
    # Check if fig or ax arguments are passed
    if fig_complex is not None:
        hc.fig_complex = fig_complex
    if ax_complex is not None:
        hc.ax_complex = ax_complex
    if fig_surface is not None:
        hc.fig_surface = fig_surface
    if ax_surface is not None:
        hc.ax_surface = ax_surface

    # Create pyplot.figure instance if none exists yet
    try:
        hc.fig_complex
    except AttributeError:
        hc.fig_complex = pyplot.figure()

    # Clear existing axes so previous plot elements do not persist
    try:
        hc.ax_complex.cla()
    except AttributeError:
        pass
    try:
        hc.ax_surface.cla()
    except AttributeError:
        pass

    # Consistency
    if hc.sfield is None:
        if contour_plot:
            contour_plot = False
            logging.warning("Warning, no associated scalar field found. "
                            "Not plotting contour_plot.")

        if surface_field_plot:
            surface_field_plot = False
            logging.warning("Warning, no associated scalar field found. "
                            "Not plotting surface field.")

    # Define colours:
    coldict = {'lo': numpy.array([242, 189, 138]) / 255,  # light orange
               'do': numpy.array([235, 129, 27]) / 255  # Dark alert orange
               }

    def define_cols(col):
        if (col == 'lo') or (col == 'do'):
            col = coldict[col]
        elif col is None:
            col = None
        return col

    point_color = define_cols(point_color)  # None will generate
    line_color = define_cols(line_color)
    complex_color_f = define_cols(complex_color_f)
    complex_color_e = define_cols(complex_color_e)

    if hc.dim == 1:
        if arrow_width is not None:
            hc.arrow_width = arrow_width
            hc.mutation_scale = 58.83484054145521 * hc.arrow_width * 1.3
        else:  # heuristic
            dx = hc.bounds[0][1] - hc.bounds[0][0]
            hc.arrow_width = (dx * 0.13
                                / (numpy.sqrt(len(hc.V.cache))))
            hc.mutation_scale = 58.83484054145521 * hc.arrow_width * 1.3

        try:
            hc.ax_complex
        except:
            hc.ax_complex = hc.fig_complex.add_subplot(1, 1, 1)

        min_points = []
        for v in hc.V.cache:
            hc.ax_complex.plot(v, 0, '.',
                                 color=point_color,
                                 markersize=pointsize)
            xlines = []
            ylines = []
            for v2 in hc.V[v].nn:
                xlines.append(v2.x)
                ylines.append(0)

                if directed:
                    if hc.V[v].f > v2.f:  # direct V2 --> V1
                        x1_vec = list(hc.V[v].x)
                        x2_vec = list(v2.x)
                        x1_vec.append(0)
                        x2_vec.append(0)
                        ap = plot_directed_edge(hc.V[v].f, v2.f,
                                                     x1_vec, x2_vec,
                                                     mut_scale=0.5 * hc.mutation_scale,
                                                     proj_dim=2,
                                                     color=line_color)

                        hc.ax_complex.add_patch(ap)

            if directed:
                if minimiser_points:
                    if hc.V[v].minimiser():
                        v_min = list(v)
                        v_min.append(0)
                        min_points.append(v_min)

            hc.ax_complex.plot(xlines, ylines, color=line_color)

        if minimiser_points:
            hc.ax_complex = plot_min_points(hc.ax_complex,
                                                   min_points,
                                                   proj_dim=2,
                                                   point_color=point_color,
                                                   pointsize=pointsize)

        # Clean up figure
        if hc.bounds is None:
            pyplot.ylim([-1e-2, 1 + 1e-2])
            pyplot.xlim([-1e-2, 1 + 1e-2])
        else:
            fac = 1e-2  # TODO: TEST THIS
            pyplot.ylim([0 - fac, 0 + fac])
            pyplot.xlim(
                [hc.bounds[0][0] - fac * (hc.bounds[0][1]
                                            - hc.bounds[0][0]),
                 hc.bounds[0][1] + fac * (hc.bounds[0][1]
                                            - hc.bounds[0][0])])
        if no_grids:
            hc.ax_complex.set_xticks([])
            hc.ax_complex.set_yticks([])
            hc.ax_complex.axis('off')

        # Surface plots
        if surface_plot or surface_field_plot:
            try:
                hc.fig_surface
            except AttributeError:
                hc.fig_surface = pyplot.figure()
            try:
                hc.ax_surface
            except:
                hc.ax_surface = hc.fig_surface.add_subplot(1, 1, 1)

            # Add a plot of the field function.
            if surface_field_plot:
                hc.fig_surface, hc.ax_surface = plot_field_surface(
                    hc,
                    hc.fig_surface,
                    hc.ax_surface,
                    hc.bounds,
                    hc.sfield,
                    hc.sfield_args,
                    proj_dim=2,
                    color=complex_color_f)  # TODO: Custom field colour

            if surface_plot:
                hc.fig_surface, hc.ax_surface = plot_complex_surface(
                    hc,
                    hc.fig_surface,
                    hc.ax_surface,
                    directed=directed,
                    pointsize=pointsize,
                    color_e=complex_color_e,
                    color_f=complex_color_f,
                    min_points=min_points)

            if no_grids:
                hc.ax_surface.set_xticks([])
                hc.ax_surface.set_yticks([])
                hc.ax_surface.axis('off')

    elif hc.dim == 2:
        if arrow_width is not None:
            hc.arrow_width = arrow_width
        else:  # heuristic
            dx1 = hc.bounds[0][1] - hc.bounds[0][0]
            dx2 = hc.bounds[1][1] - hc.bounds[1][0]

            try:
                hc.arrow_width = (min(dx1, dx2) * 0.13
                                    / (numpy.sqrt(len(hc.V.cache))))
            except TypeError:  # Allow for decimal operations
                hc.arrow_width = (min(dx1, dx2) * decimal.Decimal(0.13)
                                    / decimal.Decimal(
                            (numpy.sqrt(len(hc.V.cache)))))

        try:
            hc.mutation_scale = 58.8348 * hc.arrow_width * 1.5
        except TypeError:  # Allow for decimal operations
            hc.mutation_scale = (decimal.Decimal(58.8348)
                                   * hc.arrow_width
                                   * decimal.Decimal(1.5))

        try:
            hc.ax_complex
        except:
            hc.ax_complex = hc.fig_complex.add_subplot(1, 1, 1)

        if contour_plot:
            plot_contour(hc, hc.bounds, hc.sfield,
                              hc.sfield_args)

        if complex_plot:
            min_points = []
            for v in hc.V.cache:
                hc.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                     markersize=pointsize)

                xlines = []
                ylines = []
                for v2 in hc.V[v].nn:
                    xlines.append(v2.x[0])
                    ylines.append(v2.x[1])
                    xlines.append(v[0])
                    ylines.append(v[1])

                    if directed:
                        if hc.V[v].f > v2.f:  # direct V2 --> V1
                            ap = plot_directed_edge(hc.V[v].f, v2.f,
                                                         hc.V[v].x, v2.x,
                                                         mut_scale=hc.mutation_scale,
                                                         proj_dim=2,
                                                         color=line_color)

                            hc.ax_complex.add_patch(ap)
                if directed:
                    if minimiser_points:
                        if hc.V[v].minimiser():
                            min_points.append(v)

                hc.ax_complex.plot(xlines, ylines, color=line_color)

            if directed:
                if minimiser_points:
                    hc.ax_complex = plot_min_points(hc.ax_complex,
                                                           min_points,
                                                           proj_dim=2,
                                                           point_color=point_color,
                                                           pointsize=pointsize)
            else:
                min_points = []

        # Clean up figure
        if hc.bounds is None:
            pyplot.ylim([-1e-2, 1 + 1e-2])
            pyplot.xlim([-1e-2, 1 + 1e-2])
        else:
            fac = 1e-2  # TODO: TEST THIS
            pyplot.ylim(
                [hc.bounds[1][0] - fac * (hc.bounds[1][1]
                                            - hc.bounds[1][0]),
                 hc.bounds[1][1] + fac * (hc.bounds[1][1]
                                            - hc.bounds[1][0])])
            pyplot.xlim(
                [hc.bounds[0][0] - fac * (hc.bounds[1][1]
                                            - hc.bounds[1][0]),
                 hc.bounds[0][1] + fac * (hc.bounds[1][1]
                                            - hc.bounds[1][0])])

        if no_grids:
            hc.ax_complex.set_xticks([])
            hc.ax_complex.set_yticks([])
            hc.ax_complex.axis('off')

        # Surface plots
        if surface_plot or surface_field_plot:
            try:
                hc.fig_surface
            except AttributeError:
                hc.fig_surface = pyplot.figure()
            try:
                hc.ax_surface
            except:
                hc.ax_surface = hc.fig_surface.add_subplot(projection='3d')

            # Add a plot of the field function.
            if surface_field_plot:
                hc.fig_surface, hc.ax_surface = plot_field_surface(
                    hc,
                    hc.fig_surface,
                    hc.ax_surface,
                    hc.bounds,
                    hc.sfield,
                    hc.sfield_args,
                    proj_dim=3)

            if surface_plot:
                hc.fig_surface, hc.ax_surface = plot_complex_surface(
                    hc,
                    hc.fig_surface,
                    hc.ax_surface,
                    directed=directed,
                    pointsize=pointsize,
                    color_e=complex_color_e,
                    color_f=complex_color_f,
                    min_points=min_points)

            if no_grids:
                hc.ax_surface.set_xticks([])
                hc.ax_surface.set_yticks([])
                hc.ax_surface.axis('off')


    elif hc.dim == 3:
        #try:
        #    hc.ax_complex
        #except:
        #    hc.ax_complex = Axes3D(hc.fig_complex)

        try:
            hc.fig_complex
        except AttributeError:
            hc.fig_complex = pyplot.figure()
        try:
            hc.ax_complex
        except:
            hc.ax_complex = hc.fig_complex.add_subplot(projection='3d')

        min_points = []
        for v in hc.V.cache:
            hc.ax_complex.scatter(v[0], v[1], v[2],
                                    color=point_color, s=pointsize)
            x = []
            y = []
            z = []
            x.append(hc.V[v].x[0])
            y.append(hc.V[v].x[1])
            z.append(hc.V[v].x[2])
            for v2 in hc.V[v].nn:
                x.append(v2.x[0])
                y.append(v2.x[1])
                z.append(v2.x[2])
                x.append(hc.V[v].x[0])
                y.append(hc.V[v].x[1])
                z.append(hc.V[v].x[2])
                if directed:
                    if hc.V[v].f > v2.f:  # direct V2 --> V1
                        ap = plot_directed_edge(hc.V[v].f, v2.f,
                                                     hc.V[v].x, v2.x,
                                                     proj_dim=3,
                                                     color=line_color)
                        hc.ax_complex.add_artist(ap)

            hc.ax_complex.plot(x, y, z,
                                 color=line_color)
            if directed:
                if minimiser_points:
                    if hc.V[v].minimiser():
                        min_points.append(v)

        if minimiser_points:
            hc.ax_complex = plot_min_points(hc.ax_complex,
                                                   min_points,
                                                   proj_dim=3,
                                                   point_color=point_color,
                                                   pointsize=pointsize)

        hc.fig_surface = None  # Current default
        hc.ax_surface = None  # Current default

    else:
        logging.warning("dimension higher than 3 or wrong complex format")
        hc.fig_complex = None
        hc.ax_complex = None
        hc.fig_surface = None
        hc.ax_surface = None

    # Save figure to file
    if save_fig:
        if strpath is None:
            script_dir = os.getcwd()  # os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, plot_path)
            sample_file_name = fig_name

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            strpath = results_dir + sample_file_name

        plot_save_figure(hc, strpath)


    if show and (not hc.dim > 3):
        hc.fig_complex.show()
    try:
        hc.fig_surface
        hc.ax_surface
        if show:
            hc.fig_surface.show()
    except AttributeError:
        hc.fig_surface = None  # Set to None for return reference
        hc.ax_surface = None

    return hc.fig_complex, hc.ax_complex, hc.fig_surface, hc.ax_surface


def plot_save_figure(hc, strpath):

    hc.fig_complex.savefig(strpath, transparent=True,
                             bbox_inches='tight', pad_inches=0)


def plot_clean(hc, del_ax=True, del_fig=True):
    try:
        if del_ax:
            del (hc.ax_complex)
        if del_fig:
            del (hc.fig_complex)
    except AttributeError:
        pass


def plot_contour(hc, bounds, func, func_args=()):
    """
    Plots the field functions. Mostly for developmental purposes
    :param hc: Complex instance
    :param bounds:
    :param func:
    :param func_args:
    :param surface:
    :param contour:
    :return:
    """
    xg, yg, Z = plot_field_grids(hc, bounds, func, func_args)
    cs = pyplot.contour(xg, yg, Z, cmap='binary_r', color='k')
    pyplot.clabel(cs)


def plot_complex_surface(hc, fig, ax, directed=True, pointsize=5,
                         color_e=None, color_f=None, min_points=[]):
    """
    fig and ax need to be supplied outside the method
    :param hc: Complex instance
    :param fig: ex. ```fig = pyplot.figure()```
    :param ax: ex.  ```ax = fig.gca(projection='3d')```
    :param bounds:
    :param func:
    :param func_args:
    :return:
    """
    if hc.dim == 1:
        # Plot edges
        z = []
        for v in hc.V.cache:
            if directed:
                ax.plot(v, hc.V[v].f, '.', color=color_e,
                        markersize=pointsize)
                z.append(hc.V[v].f)
            else:
                ax.plot(v, 0.0, '.', color=color_e,
                        markersize=pointsize)
                z.append(0.0)

            for v2 in hc.V[v].nn:
                if directed:
                    ax.plot([v, v2.x],
                            [hc.V[v].f, v2.f],
                            color=color_e)
                    if hc.V[v].f > v2.f:  # direct V2 --> V1
                        x1_vec = [float(hc.V[v].x[0]), float(hc.V[v].f)]
                        x2_vec = [float(v2.x[0]), float(v2.f)]

                        a = plot_directed_edge(hc.V[v].f, v2.f,
                                                    x1_vec, x2_vec,
                                                    proj_dim=2,
                                                    color=color_e)
                        ax.add_artist(a)
                else:
                    ax.plot([v, v2.x],
                            [0.0, 0.0],
                            color=color_e)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$f$')

        if len(min_points) > 0:
            iter_min = min_points.copy()
            for ind, v in enumerate(iter_min):
                min_points[ind][1] = float(hc.V[v[0],].f)

            ax = plot_min_points(ax,
                                      min_points,
                                      proj_dim=2,
                                      point_color=color_e,
                                      pointsize=pointsize
                                      )

    elif hc.dim == 2:
        # Plot edges
        z = []
        for v in hc.V.cache:
            if directed:
                z.append(hc.V[v].f)
            else:
                z.append(0.0)
            for v2 in hc.V[v].nn:
                if directed:
                    ax.plot([v[0], v2.x[0]],
                            [v[1], v2.x[1]],
                            [hc.V[v].f, v2.f],
                            color=color_e)

                    if hc.V[v].f > v2.f:  # direct V2 --> V1
                        x1_vec = list(hc.V[v].x)
                        x2_vec = list(v2.x)
                        x1_vec.append(hc.V[v].f)
                        x2_vec.append(v2.f)
                        a = plot_directed_edge(hc.V[v].f, v2.f,
                                                    x1_vec, x2_vec,
                                                    proj_dim=3,
                                                    color=color_e)

                        ax.add_artist(a)
                else:
                    ax.plot([v[0], v2.x[0]],
                            [v[1], v2.x[1]],
                            [0.0, 0.0],
                            color=color_e)

        # TODO: For some reason adding the scatterplots for minimiser spheres
        #      makes the directed edges disappear behind the field surface
        if directed:
            if len(min_points) > 0:
                iter_min = min_points.copy()
                for ind, v in enumerate(iter_min):
                    min_points[ind] = list(min_points[ind])
                    min_points[ind].append(hc.V[v].f)

                ax = plot_min_points(ax,
                                          min_points,
                                          proj_dim=3,
                                          point_color=color_e,
                                          pointsize=pointsize
                                          )

        # Triangulation to plot faces
        # Compute a triangulation #NOTE: can eat memory
        hc.vertex_face_mesh()

        ax.plot_trisurf(numpy.array(hc.vertices_fm)[:, 0],
                        numpy.array(hc.vertices_fm)[:, 1],
                        z,
                        triangles=numpy.array(hc.simplices_fm_i),
                        # TODO: Select colour scheme
                        color=color_f,
                        alpha=0.4,
                        linewidth=0.2,
                        antialiased=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')

    return fig, ax


def plot_field_surface(hc, fig, ax, bounds, func, func_args=(),
                       proj_dim=2, color=None):
    """
    fig and ax need to be supplied outside the method
    :param hc: Complex instance
    :param fig: ex. ```fig = pyplot.figure()```
    :param ax: ex.  ```ax = fig.gca(projection='3d')```
    :param bounds:
    :param func:
    :param func_args:
    :return:
    """
    if proj_dim == 2:
        from matplotlib import cm
        xr = numpy.linspace(hc.bounds[0][0], hc.bounds[0][1], num=1000)
        fr = numpy.zeros_like(xr)
        for i in range(xr.shape[0]):
            fr[i] = func(xr[i], *func_args)

        ax.plot(xr, fr, alpha=0.6, color=color)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$f$')

    if proj_dim == 3:
        from matplotlib import cm
        xg, yg, Z = plot_field_grids(hc, bounds, func, func_args)
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        # cmap=cm.coolwarm,
                        # cmap=cm.magma,
                    #    cmap=cm.plasma,  #TODO: Restore
                        # cmap=cm.inferno,
                        # cmap=cm.pink,
                        # cmap=cm.viridis,
                        #facecolors="do it differently, ok?",
                        color = [0.94901961, 0.74117647, 0.54117647],
                        linewidth=0,
                        antialiased=True, alpha=0.8, shade=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')
    return fig, ax


def plot_field_grids(hc, bounds, func, func_args):
    try:
        return hc.plot_xg, hc.plot_yg, hc.plot_Z
    except AttributeError:
        X = numpy.linspace(bounds[0][0], bounds[0][1])
        Y = numpy.linspace(bounds[1][0], bounds[1][1])
        xg, yg = numpy.meshgrid(X, Y)
        Z = numpy.zeros((xg.shape[0],
                         yg.shape[0]))

        for i in range(xg.shape[0]):
            for j in range(yg.shape[0]):
                Z[i, j] = func(numpy.array([xg[i, j], yg[i, j]]),
                               *func_args)

        hc.plot_xg, hc.plot_yg, hc.plot_Z = xg, yg, Z
        return hc.plot_xg, hc.plot_yg, hc.plot_Z


def plot_directed_edge(f_v1, f_v2, x_v1, x_v2, mut_scale=20,
                       proj_dim=2,
                       color=None):
    """
    Draw a directed edge embeded in 2 or 3 dimensional space between two
    vertices v1 and v2.

    :param f_v1: field value at f(v1)
    :param f_v2: field value at f(v2)
    :param x_v1: coordinate vector 1
    :param x_v2: coordinate vector 2
    :param proj_dim: int, must be either 2 or 3
    :param color: edge color
    :return: a, artist arrow object (add with ex. Axes.add_artist(a)
    """
    if proj_dim == 2:
        if f_v1 > f_v2:  # direct V2 --> V1
            dV = numpy.array(x_v1) - numpy.array(x_v2)
            ap = matplotlib.patches.FancyArrowPatch(
                numpy.array(x_v2) + 0.5 * dV,  # tail
                numpy.array(x_v2) + 0.6 * dV,  # head
                mutation_scale=mut_scale,
                arrowstyle='-|>',
                fc=color, ec=color,
                color=color,
            )

    if proj_dim == 3:
        if f_v1 > f_v2:  # direct V2 --> V1
            dV = numpy.array(x_v1) - numpy.array(x_v2)
            # TODO: Might not be correct (unvalidated)
            ap = Arrow3D([x_v2[0], x_v2[0] + 0.5 * dV[0]],
                         [x_v2[1], x_v2[1] + 0.5 * dV[1]],
                         [x_v2[2], x_v2[2] + 0.5 * dV[2]],
                         mutation_scale=20,
                         lw=1, arrowstyle="-|>",
                         color=color)

    return ap


def plot_min_points(axes, min_points, proj_dim=2, point_color=None,
                    pointsize=5):
    """
    Add a given list of highlighted minimiser points to axes

    :param ax: An initiated matplotlib Axes class
    :param min_points: list of minimsier points
    :param proj_dim: projection dimension, must be either 2 or 3
    :param point_color: optional
    :param point_size: optional
    :return:
    """
    is_red = isinstance(point_color, str) and point_color == 'r'

    if proj_dim == 2:
        for v in min_points:
            min_col = 'k' if is_red else 'r'

            axes.plot(v[0], v[1], '.', color=point_color,
                      markersize=2.5 * pointsize)

            axes.plot(v[0], v[1], '.', color='k',
                      markersize=1.5 * pointsize)

            axes.plot(v[0], v[1], '.', color=min_col,
                      markersize=1.4 * pointsize)

    if proj_dim == 3:
        for v in min_points:
            min_col = 'k' if is_red else 'r'

            axes.scatter(v[0], v[1], v[2], color=point_color,
                         s=2.5 * pointsize)

            axes.scatter(v[0], v[1], v[2], color='k',
                         s=1.5 * pointsize)

            axes.scatter(v[0], v[1], v[2], color=min_col,
                         s=1.4 * pointsize)

    return axes


def animate_complex(hc, update_state, frames=200, interval=50,
                    repeat=True, save_path=None, fps=20,
                    figsize=None, **plot_kwargs):
    """Animate a simplicial complex by re-rendering with ``plot_complex``.

    Each frame calls *update_state* to mutate the Complex in-place (move,
    add or remove vertices, connect or disconnect edges), then redraws the
    mesh using :func:`plot_complex` so the visual style is identical.

    Parameters
    ----------
    hc : Complex
        A triangulated Complex instance (1-D, 2-D or 3-D).
    update_state : callable
        ``update_state(hc, frame) -> None``
        Called once per frame **before** rendering.  Should mutate *hc*
        in-place, for example via ``hc.V.move(v, new_x)``,
        ``hc.V.remove(v)``, ``v.connect(v2)`` / ``v.disconnect(v2)``.
    frames : int
        Number of animation frames.
    interval : int
        Delay between frames in milliseconds.
    repeat : bool
        Whether the animation loops.
    save_path : str, optional
        If given, save the animation to this file (e.g. ``"mesh.gif"``).
    fps : int
        Frames per second when saving.
    figsize : tuple, optional
        Figure size passed to ``pyplot.figure``.
    **plot_kwargs
        Additional keyword arguments forwarded to :func:`plot_complex`
        (e.g. ``point_color``, ``pointsize``, ``line_color``,
        ``no_grids``).  ``show`` and ``save_fig`` are always overridden
        to ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    anim : matplotlib.animation.FuncAnimation
    """
    if not matplotlib_available:
        logging.warning("matplotlib is required for animate_complex")
        return None, None, None

    from matplotlib.animation import FuncAnimation

    # Force non-interactive rendering for animation frames
    plot_kwargs['show'] = False
    plot_kwargs['save_fig'] = False
    plot_kwargs.setdefault('directed', False)
    # Suppress surface/contour plots by default (often no scalar field)
    plot_kwargs.setdefault('contour_plot', False)
    plot_kwargs.setdefault('surface_plot', False)
    plot_kwargs.setdefault('surface_field_plot', False)

    # Create figure and axes, store on hc so plot_complex reuses them
    fig = pyplot.figure(figsize=figsize)
    if hc.dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1)

    hc.fig_complex = fig
    hc.ax_complex = ax

    # Render the initial frame
    plot_complex(hc, **plot_kwargs)

    def _update(frame):
        update_state(hc, frame)
        plot_complex(hc, **plot_kwargs)

    anim = FuncAnimation(fig, _update, frames=frames, interval=interval,
                         blit=False, repeat=repeat)

    if save_path:
        anim.save(save_path, fps=fps)

    return fig, ax, anim
