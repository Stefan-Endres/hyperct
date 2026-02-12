"""
Visualization utilities for DDG dual meshes.

Matplotlib-based plotting for 1D, 2D, and 3D primal/dual mesh pairs.
Blue = primal mesh, Orange = dual mesh, Green dashed = primal-to-dual
connections.
"""
from __future__ import annotations

import numpy as np


def plot_dual_mesh_1D(HC, ax=None, show: bool = True):
    """Plot a 1D complex and its dual mesh (midpoints).

    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param ax: Optional matplotlib Axes. Created if None.
    :param show: If True, call ``plt.show()``.
    :return: (fig, ax) tuple.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    else:
        fig = ax.get_figure()

    # Plot primal edges
    plotted_edges = set()
    for v in HC.V:
        for v2 in v.nn:
            edge = tuple(sorted([v.x, v2.x]))
            if edge not in plotted_edges:
                ax.plot(
                    [v.x[0], v2.x[0]], [0, 0],
                    '-', color='tab:blue', linewidth=2,
                )
                plotted_edges.add(edge)

    # Plot primal vertices
    primal_x = [v.x[0] for v in HC.V]
    ax.plot(primal_x, [0] * len(primal_x), 'o', color='tab:blue',
            markersize=8, label='Primal')

    # Plot dual vertices
    dual_x = [vd.x[0] for vd in HC.Vd]
    ax.plot(dual_x, [0] * len(dual_x), 's', color='tab:orange',
            markersize=6, label='Dual')

    # Connect primal to dual (dashed green)
    for v in HC.V:
        for vd in v.vd:
            ax.plot(
                [v.x[0], vd.x[0]], [0, 0],
                '--', color='tab:green', alpha=0.5, linewidth=0.8,
            )

    ax.set_yticks([])
    ax.legend()
    ax.set_title('1D Primal (blue) + Dual (orange)')

    if show:
        plt.show()
    return fig, ax


def plot_dual_mesh_2D(HC, ax=None, show: bool = True):
    """Plot a 2D primal mesh and its dual mesh.

    Blue = primal edges/vertices, Orange = dual edges/vertices,
    Green dashed = primal-to-dual connections.

    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param ax: Optional matplotlib Axes. Created if None.
    :param show: If True, call ``plt.show()``.
    :return: (fig, ax) tuple.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Plot primal edges
    plotted_edges = set()
    for v in HC.V:
        for v2 in v.nn:
            edge = tuple(sorted([v.x, v2.x]))
            if edge not in plotted_edges:
                ax.plot(
                    [v.x[0], v2.x[0]], [v.x[1], v2.x[1]],
                    '-', color='tab:blue', linewidth=1.0,
                )
                plotted_edges.add(edge)

    # Plot primal vertices
    primal_pts = np.array([v.x_a for v in HC.V])
    ax.plot(primal_pts[:, 0], primal_pts[:, 1], 'o', color='tab:blue',
            markersize=6, label='Primal')

    # Plot dual edges (connect shared dual vertices between primal neighbours)
    plotted_dual_edges = set()
    for v in HC.V:
        for v2 in v.nn:
            shared = v.vd.intersection(v2.vd)
            if len(shared) < 2:
                continue
            shared_list = list(shared)
            for i in range(len(shared_list)):
                for j in range(i + 1, len(shared_list)):
                    vd1, vd2 = shared_list[i], shared_list[j]
                    de = tuple(sorted([vd1.x, vd2.x]))
                    if de not in plotted_dual_edges:
                        ax.plot(
                            [vd1.x[0], vd2.x[0]], [vd1.x[1], vd2.x[1]],
                            '-', color='tab:orange', linewidth=1.5,
                        )
                        plotted_dual_edges.add(de)

    # Plot dual vertices
    dual_pts = np.array([vd.x_a for vd in HC.Vd])
    ax.plot(dual_pts[:, 0], dual_pts[:, 1], 'o', color='tab:orange',
            markersize=4, label='Dual')

    # Connect primal to dual (dashed green)
    for v in HC.V:
        for vd in v.vd:
            ax.plot(
                [v.x[0], vd.x[0]], [v.x[1], vd.x[1]],
                '--', color='tab:green', alpha=0.3, linewidth=0.5,
            )

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('2D Primal (blue) + Dual (orange)')

    if show:
        plt.show()
    return fig, ax


def plot_dual_mesh_3D(HC, ax=None, show: bool = True):
    """Plot a 3D primal mesh and its dual vertices.

    :param HC: Complex with ``HC.Vd`` populated by ``compute_vd``.
    :param ax: Optional matplotlib 3D Axes. Created if None.
    :param show: If True, call ``plt.show()``.
    :return: (fig, ax) tuple.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot primal edges
    plotted_edges = set()
    for v in HC.V:
        for v2 in v.nn:
            edge = tuple(sorted([v.x, v2.x]))
            if edge not in plotted_edges:
                ax.plot(
                    [v.x[0], v2.x[0]],
                    [v.x[1], v2.x[1]],
                    [v.x[2], v2.x[2]],
                    '-', color='tab:blue', linewidth=0.8,
                )
                plotted_edges.add(edge)

    # Plot primal vertices
    primal_pts = np.array([v.x_a for v in HC.V])
    ax.scatter(
        primal_pts[:, 0], primal_pts[:, 1], primal_pts[:, 2],
        color='tab:blue', s=20, label='Primal',
    )

    # Plot dual vertices
    dual_pts = np.array([vd.x_a for vd in HC.Vd])
    ax.scatter(
        dual_pts[:, 0], dual_pts[:, 1], dual_pts[:, 2],
        color='tab:orange', s=10, label='Dual',
    )

    # Connect primal to dual (dashed green)
    for v in HC.V:
        for vd in v.vd:
            ax.plot(
                [v.x[0], vd.x[0]],
                [v.x[1], vd.x[1]],
                [v.x[2], vd.x[2]],
                '--', color='tab:green', alpha=0.3, linewidth=0.5,
            )

    ax.legend()
    ax.set_title('3D Primal (blue) + Dual (orange)')

    if show:
        plt.show()
    return fig, ax
