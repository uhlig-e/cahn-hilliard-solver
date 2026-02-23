from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from chsol_v3 import chsol
from ch_solver_v3 import do_solve
import pickle
from matplotlib.lines import Line2D

from matplotlib.colors import LinearSegmentedColormap


'''
import psutil

process = psutil.Process()
init_core = 0
num_cores = 4
cores = list(np.arrange(init_core, init_core + num_cores))
process.cpu_affinity(cores)
'''

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

def im(file: chsol, phi_l=-1, phi_r=1.5, wave=False, step=False, k=0.01, a=0.01) -> chsol:
    fs = file.make_function_space()
    fss = fs.sub(0).collapse()
    x0 = 0
    Ly = file.params['Ly']
    l = file.ell
    psi_expr = Expression('0.0', degree=1)

    if not step and not wave:
        phi_expr = Expression(
        '0.5*(a + b) + 0.5*(b - a)*tanh((x[0] - x0)/l)',
        a=phi_l, b=phi_r, x0=x0, l=l,
        degree=3
        )

    if wave:
        phi_expr = Expression(
            '0.5*(a + b) + 0.5*(b - a)*tanh((x[0] - (x0 + c * sin(2 * pi * k * x[1])))/l)', 
            degree=4, x0=x0, a=phi_l, b=phi_r, k=k, l=l, c=0.01, Ly=Ly)
    
    if step and not wave:
        phi_expr = Expression('x[0] < x0 ? phi_left : phi_right', degree=1, x0=x0, phi_left=phi_l, phi_right=phi_r)

    if step and wave:
        phi_expr = Expression('x[0] < x0 + a*sin(2*pi*k*x[1]) ? phi_left: phi_right + a * exp(-(x[0] - x0 - a*sin(2*pi*k*x[1])))', degree=1, a=a, k=k, phi_left=phi_l, phi_right=phi_r, x0=x0)
        phi_expr = Expression('x[0] < x0 + a*sin(2*pi*k*x[1]) ? phi_left: phi_right', degree=1, a=a, k=k, phi_left=phi_l, phi_right=phi_r, x0=x0) # no exp here...

    phi0, psi0 = Function(fss), Function(fss)
    phi0.interpolate(phi_expr)
    psi0.interpolate(psi_expr)
    assigner = FunctionAssigner(fs, [fss, fss])
    v = Function(fs)
    assign(v, [phi0, psi0])
    return v

ks = [1e-2]#, 2e-2, 5e-2, 7e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1, 2, 5, 7, 10, 12, 15, 17]
for k in ks:
    eps = 0.1
    Lx = 50 # simulation box x-length
    wavelength = int(1/k)
    no_waves = 3
    Ly = int(no_waves * wavelength)
    nodes_per_wave = 10
    init_x_points_per_side = 40
    init_y_points = nodes_per_wave * no_waves
    deg = 1
    t_f = 10
    dt = 1e-2
    phi_r = 0.6
    amp = 0.05

    params = {
        'eps': eps,
        'Lx': Lx,
        'Ly': Ly,
        'n_x': init_x_points_per_side,
        'n_y': init_y_points,
        'deg': deg,
        't_f': t_f,
        'dt': dt,
        'k': k,
        'solver_params': {'min_dt': 1e-8, 'safety_factor':0.8, 'max_newton_iter': 25}
    }
    fn = 'v3test'
    obj = chsol(fn, params)
    x0=0
    init_func = im(obj, phi_r=phi_r, step=True, wave=True, k=k, a=amp)
    # --------------------------------------------------
    # 1. INITIAL MESH REPRESENTATION
    # --------------------------------------------------
    init_mesh = obj.get_mesh()
    coords_init = init_mesh.coordinates()
    order_init = np.lexsort((coords_init[:, 0], coords_init[:, 1]))

    logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=init_x_points_per_side)
    logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=init_x_points_per_side)
    x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])

    # Now rebuild the actual x,y from mesh
    x_unique = np.unique(coords_init[:,0])
    y_unique = np.unique(coords_init[:,1])

    X, Y = np.meshgrid(x_unique, y_unique)
    y = np.linspace(-Ly/2, Ly/2, num=init_y_points)

    ny_init = len(y)
    nx_init = len(x)

    phi_init_vals = init_func.compute_vertex_values(init_mesh)[order_init]
    phi_init_grid = phi_init_vals.reshape(ny_init, nx_init)


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
    obj = do_solve(init_func, obj, boundary_key='dirichlet', bdval=phi_r, track_interface=True)
    obj.save()
    obj.plot_mode_amp()
    print(obj.k_amps)
    obj.plot_dendrite()
    obj.make_movie()




'''
fn = 'a_0.05_phi_r_0.6_eps_0.1_k_{}'.format(k)
obj = chsol(fn, params)
init_func = im(obj, phi_r=phi_r, step=True, wave=True, k=k, a=amp)
obj = do_solve(init_func, obj, boundary_key='dirichlet', bdval=phi_r, track_interface=True)
obj.save()
#obj.make_movie(bounds=(5, 1/k), interface_positions=obj.interface_positions)
'''