from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os

class ComplexPlotter:
    def __init__(self, V, dim, domain=None, sfield=None, sfield_args=(),
                 vfield=None, vfield_args=None, g_cons=None, g_cons_args=()):
        self.V = V
        self.dim = dim
        self.domain = domain
        if domain is None:
            self.bounds = [(0, 1),]*dim
        else:
            self.bounds = domain

        # Field functions
        self.sfield = sfield
        self.vfield = vfield
        self.sfield_args = sfield_args
        self.vfield_args = vfield_args

        # Constraint functions
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def plot_complex(self):
        """
             Here C is the LIST of simplexes S in the
             2 or 3 dimensional complex

             To plot a single simplex S in a set C, use ex. [C[0]]
        """

        # Create pyplot.figure instance if none exists yet
        try:
            self.fig_complex
        except AttributeError:
            self.fig_complex = pyplot.figure()

        # Define colours:
        lo = numpy.array([242, 189, 138]) / 255  # light orange
        do = numpy.array([235, 129, 27]) / 255  # Dark alert orange

        directed = True
        contour_plot = True
        surface_plot = True  # A 3 dimensional surface + contour plot
        surface_field_plot = 0  # True  # A 3 dimensional surface + contour plot
        minimiser_points = True
        # TODO: Add dict for visual parameters
        point_color = do  # None will generate
        line_color = do
        pointsize = 5
        no_grids = False
        save_fig = True
        strpath = None  # Full string path of the file name
        plot_path = 'fig/'  # Name of the relative directory to save
        fig_name = 'complex.pdf'  # Name of the complex file to save
        arrow_width = None

        if arrow_width is not None:
            self.arrow_width = arrow_width
        else:  # heuristic
            dx1 = self.bounds[0][1] - self.bounds[0][0]
            dx2 = self.bounds[1][1] - self.bounds[1][0]
            numpy.linalg.norm([dx1, dx2])
            self.arrow_width = (numpy.linalg.norm([dx1, dx2]) * 0.13
                                # * 0.1600781059358212
                                / (numpy.sqrt(len(self.V.cache))))
            print(self.arrow_width)

        lw = 1  # linewidth

        if self.dim == 1:
            pass  # TODO: IMPLEMENT
        if self.dim == 2:
            try:
                self.ax_complex
            except:
                self.ax_complex = self.fig_complex.add_subplot(1, 1, 1)

            if contour_plot:
                self.plot_contour(self.bounds, self.sfield,
                                  self.sfield_args)

            min_points = []
            for v in self.V.cache:
                self.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                     markersize=pointsize)

                xlines = []
                ylines = []
                for v2 in self.V[v].nn:
                    xlines.append(v2.x[0])
                    ylines.append(v2.x[1])
                    xlines.append(v[0])
                    ylines.append(v[1])

                    if directed:
                        # TODO: These arrows look ugly when the domain rectangle
                        # is too stretched. We need to define our own arrow
                        # object that draws a triangle object at the desired
                        # vector and adds it to ax_complex
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            dV = numpy.array(self.V[v].x) - numpy.array(v2.x)
                            if 1:
                                self.ax_complex.arrow(v2.x[0],
                                                      v2.x[1],
                                                      0.5 * dV[0], 0.5 * dV[1],
                                                      head_width=self.arrow_width,
                                                      head_length=self.arrow_width,
                                                      fc=line_color, ec=line_color,
                                                      color=line_color)

                            if 0:
                                self.ax_complex.annotate("",
                                                         # xy=(v2.x[0], v2.x[1]),
                                                         xy=(v2.x[0] + 0.5 * dV[0],
                                                             v2.x[1] + 0.5 * dV[1]),

                                                         xytext=(
                                                         v2.x[0] + 1 * dV[0],
                                                         v2.x[1] + 1 * dV[1]),
                                                         # xytext=(v2.x[0] ,
                                                         #        v2.x[1] ),
                                                         # xytext=(0.5 * dV[0], 0.5 * dV[1]),

                                                         arrowprops=dict(
                                                             # arrowstyle='fancy',
                                                             headwidth=self.arrow_width,
                                                             headlength=self.arrow_width,
                                                             # fc=line_color, ec=line_color,
                                                             lw=0.0000000001,
                                                             # TODO make 0
                                                             color=line_color))

                if minimiser_points:
                    if self.V[v].minimiser():
                        min_points.append(v)

                self.ax_complex.plot(xlines, ylines, color=line_color)

            if minimiser_points:
                for v in min_points:
                    if point_color is 'r':
                        min_col = 'k'
                    else:
                        min_col = 'r'

                    self.ax_complex.plot(v[0], v[1], '.', color=point_color,
                                         markersize=2.5 * pointsize)

                    self.ax_complex.plot(v[0], v[1], '.', color='k',
                                         markersize=1.5 * pointsize)

                    self.ax_complex.plot(v[0], v[1], '.', color=min_col,
                                         markersize=1.4 * pointsize)

            # Clean up figure
            if self.bounds is None:
                pyplot.ylim([-1e-2, 1 + 1e-2])
                pyplot.xlim([-1e-2, 1 + 1e-2])
            else:
                fac = 1e-2  # TODO: TEST THIS
                pyplot.ylim(
                    [self.bounds[1][0] - fac * (self.bounds[1][1]
                                                - self.bounds[1][0]),
                     self.bounds[1][1] + fac * (self.bounds[1][1]
                                                - self.bounds[1][0])])
                pyplot.xlim(
                    [self.bounds[0][0] - fac * (self.bounds[1][1]
                                                - self.bounds[1][0]),
                     self.bounds[0][1] + fac * (self.bounds[1][1]
                                                - self.bounds[1][0])])

            if no_grids:
                self.ax_complex.set_xticks([])
                self.ax_complex.set_yticks([])
                self.ax_complex.axis('off')

            # Surface plots
            if surface_plot:
                try:
                    self.fig_surface
                except AttributeError:
                    self.fig_surface = pyplot.figure()
                try:
                    self.ax_surf
                except:
                    self.ax_surf = self.fig_surface.gca(projection='3d')

                self.fig_surface, self.ax_surf = self.plot_complex_surface(
                    self.fig_surface,
                    self.ax_surf)

                # Add a plot of the field function.
                if surface_field_plot:
                    self.fig_surface, self.ax_surf = self.plot_field_surface(
                        self.fig_surface,
                        self.ax_surf,
                        self.bounds,
                        self.sfield,
                        self.sfield_args)


        elif self.dim == 3:
            try:
                self.ax_complex
            except:
                self.ax_complex = Axes3D(self.fig_complex)

            min_points = []
            for v in self.V.cache:
                self.ax_complex.scatter(v[0], v[1], v[2],
                                        color=point_color, s=pointsize)
                x = []
                y = []
                z = []
                x.append(self.V[v].x[0])
                y.append(self.V[v].x[1])
                z.append(self.V[v].x[2])
                for v2 in self.V[v].nn:
                    x.append(v2.x[0])
                    y.append(v2.x[1])
                    z.append(v2.x[2])
                    x.append(self.V[v].x[0])
                    y.append(self.V[v].x[1])
                    z.append(self.V[v].x[2])
                    if directed:
                        if self.V[v].f > v2.f:  # direct V2 --> V1
                            dV = numpy.array(self.V[v].x) - numpy.array(v2.x)
                            # TODO: Might not be correct (unvalidated)
                            a = Arrow3D([v2.x[0], v2.x[0] + 0.5 * dV[0]],
                                        [v2.x[1], v2.x[1] + 0.5 * dV[1]],
                                        [v2.x[2], v2.x[2] + 0.5 * dV[2]],
                                        mutation_scale=20,
                                        lw=1, arrowstyle="-|>", color=line_color)
                            self.ax_complex.add_artist(a)

                self.ax_complex.plot(x, y, z,
                                     color=line_color,
                                     label='simplex')

                if minimiser_points:
                    if self.V[v].minimiser():
                        min_points.append(v)

            if minimiser_points:
                for v in min_points:
                    if point_color is 'r':
                        min_col = 'k'
                    else:
                        min_col = 'r'

                    self.ax_complex.scatter(v[0], v[1], v[2],
                                            color=point_color,
                                            s=2.5 * pointsize)

                    self.ax_complex.scatter(v[0], v[1], v[2], color='k',
                                            s=1.5 * pointsize)

                    self.ax_complex.scatter(v[0], v[1], v[2], color=min_col,
                                            s=1.4 * pointsize)


        else:
            print("dimension higher than 3 or wrong complex format")

        # Save figure to file
        if save_fig:
            if strpath is None:
                script_dir = os.getcwd()  # os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, plot_path)
                sample_file_name = fig_name

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                strpath = results_dir + sample_file_name

            self.plot_save_figure(strpath)

        self.fig_complex.show()
        try:
            self.fig_surface.show()
        except AttributeError:
            pass
        return self.fig_surface, self.ax_complex

    def plot_save_figure(self, strpath):

        self.fig_complex.savefig(strpath, transparent=True,
                           bbox_inches='tight', pad_inches=0)

    def plot_clean(self, del_ax=True, del_fig=True):
        try:
            if del_ax:
                del(self.ax_complex)
            if del_fig:
                del(self.fig_complex)
        except AttributeError:
            pass

    def plot_contour(self, bounds, func, func_args=()):
        """
        Plots the field functions. Mostly for developmental purposes
        :param fig:
        :param bounds:
        :param func:
        :param func_args:
        :param surface:
        :param contour:
        :return:
        """
        xg, yg, Z = self.plot_field_grids(bounds, func, func_args)
        cs = pyplot.contour(xg, yg, Z, cmap='binary_r', color='k')
        pyplot.clabel(cs)

    def plot_complex_surface(self, fig, ax):
        """
        fig and ax need to be supplied outside the method
        :param fig: ex. ```fig = pyplot.figure()```
        :param ax: ex.  ```ax = fig.gca(projection='3d')```
        :param bounds:
        :param func:
        :param func_args:
        :return:
        """

        x = []
        y = []
        z = []
        for v in self.V.cache:
            x.append(v[0])
            y.append(v[1])
            z.append(self.V[v].f)
            for v2 in self.V[v].nn:
                x.append(v2.x[0])
                y.append(v2.x[1])
                z.append(v2.f)
                # go back to starting coord
                x.append(v[0])  #TODO: is this needed?
                y.append(v[1])
                z.append(self.V[v].f)
                ax.plot([v[0], v2.x[0]],
                        [v[1], v2.x[1]],
                        [self.V[v].f, v2.f],
                        label='simplex')

        # Add trisurf plot #TODO: Does not work properly
        ax.plot_trisurf(x, y, z,
                        #TODO: Select colour scheme
                        alpha=0.4,
                        linewidth=0.2,
                        antialiased=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')
        return fig, ax

    def plot_field_surface(self, fig, ax, bounds, func, func_args=()):
        """
        fig and ax need to be supplied outside the method
        :param fig: ex. ```fig = pyplot.figure()```
        :param ax: ex.  ```ax = fig.gca(projection='3d')```
        :param bounds:
        :param func:
        :param func_args:
        :return:
        """
        from matplotlib import cm
        xg, yg, Z = self.plot_field_grids(bounds, func, func_args)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        #cmap=cm.coolwarm,
                        #cmap=cm.magma,
                        cmap=cm.plasma,
                        #cmap=cm.inferno,
                        #cmap=cm.pink,
                        #cmap=cm.viridis,
                        linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')
        #fig.show()
        return fig, ax

    def plot_field_grids(self, bounds, func, func_args):
        try:
            return self.plot_xg, self.plot_yg, self.plot_Z
        except AttributeError:
            X = numpy.linspace(bounds[0][0], bounds[0][1])
            Y = numpy.linspace(bounds[1][0], bounds[1][1])
            xg, yg = numpy.meshgrid(X, Y)
            Z = numpy.zeros((xg.shape[0],
                             yg.shape[0]))

            for i in range(xg.shape[0]):
                for j in range(yg.shape[0]):
                    Z[i, j] = func([xg[i, j], yg[i, j]], *func_args)

            self.plot_xg, self.plot_yg, self.plot_Z = xg, yg, Z
            return self.plot_xg, self.plot_yg, self.plot_Z

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    """
    Arrow used in the plotting of 3D vecotrs

    ex.
    a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)