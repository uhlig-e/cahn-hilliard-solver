from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from chsol_v3_mod import chsol
from ch_solver_v3_mod import do_solve
import pickle
from matplotlib.lines import Line2D

from matplotlib.colors import LinearSegmentedColormap

custom_cmap_phi = LinearSegmentedColormap.from_list(
    "blue-green-yellow", ["blue", "green", "yellow"]
)

matplotlib.rcParams['text.latex.preamble'] = r"""
\usepackage{amsfonts, amsmath, amssymb, amsthm, bm, cancel, upgreek}
\renewcommand{\d}{\partial} 
\newcommand{\eps}{\varepsilon} 
\newcommand{\mb}[1]{\mathbf{#1}} 
\newcommand{\til}[1]{\widetilde{#1}} 
\newcommand{\wh}[1]{\widehat{#1}} 
\newcommand{\nt}[1]{\emph{\texttt{#1}}} 
\newcommand{\der}{\text{d}} 
\newcommand{\ord}[1]{\mathcal{O}\left( #1 \right)}
"""

ks = [1e-2]#, 2e-2, 5e-2, 7e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1, 2, 5, 7, 10, 12, 15, 17]
for k in ks:
    eps = 0.1
    Lx = 100 # simulation box x-length
    wavelength = int(1/k)
    x_pts_per_side = 40
    t_f = 40
    dt = 5e-2
    amp = 0.05
    phi_r=0.9

    fn = 'v3test'
    obj = chsol(fn, eps, k, Lx, x_pts_per_side, dt=dt, t_f=t_f)
    x0=0
    obj.build_initial(phi_r=phi_r)
    # --------------------------------------------------
    # 1. INITIAL MESH REPRESENTATION
    # --------------------------------------------------

    logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=x_pts_per_side)
    logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=x_pts_per_side)
    x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])

    phi, phi_init_grid, coords = obj.phi(0, grid=True)
    obj.plot_mesh(0)
    # Now rebuild the actual x,y from mesh
    x_unique = np.unique(coords[:,0])
    y_unique = np.unique(coords[:,1])
    X, Y = np.meshgrid(x_unique, y_unique)
    y = np.linspace(-obj.Ly/2, obj.Ly/2, num=obj.ny)

    fig, ax = plt.subplots(constrained_layout=True)

    cs1 = ax.contour(
        X,
        Y,
        phi_init_grid,
        levels=[0],
        colors='blue',
        linewidths=2
    )

    plt.xlim([0, 1.1*amp])
    plt.ylim([0, int(1/k/2)])

    plt.show()
    obj = do_solve(obj, bdval=phi_r)
    obj.save()
    obj.plot_mode_amp(show=True)
    obj.plot_profiles()
    obj.plot_dendrite()
    #obj.plot_spectrum()
    #stored_val, computed_val = obj.diagnose_interface_discrepancy(10)
    #obj.make_movie()




'''
fn = 'a_0.05_phi_r_0.6_eps_0.1_k_{}'.format(k)
obj = chsol(fn, params)
init_func = im(obj, phi_r=phi_r, step=True, wave=True, k=k, a=amp)
obj = do_solve(init_func, obj, boundary_key='dirichlet', bdval=phi_r, track_interface=True)
obj.save()
#obj.make_movie(bounds=(5, 1/k), interface_positions=obj.interface_positions)
'''
