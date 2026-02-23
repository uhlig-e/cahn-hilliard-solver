from fenics import *
from chsol_v2 import chsol
from datetime import datetime
import numpy as np
from dolfin import set_log_level, LogLevel
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)

def make_bcs(file: chsol, key='', bdval=1):
    left_value = -1
    right_value = bdval 
    if key == 'dirchlet':
        fs = file.make_function_space()
        Lx = file.params['Lx']

        def left(x, on_boundary):
            return on_boundary and near(x[0], -Lx/2)

        def right(x, on_boundary):
            return on_boundary and near(x[0], Lx/2)

        bc_u_left = DirichletBC(fs.sub(0), left_value, left)
        bc_u_right = DirichletBC(fs.sub(0), right_value, right)
        bcs = [bc_u_left, bc_u_right]
    else:
        bcs = []
    return bcs


def do_solve(init_func: Function, file: chsol, boundary_key='', notes='', safe=False, bdval=1, l2=False, track_interface=False) -> chsol:
    eps = file.params['eps']
    t_f, dt = file.params['t_f'], file.params['dt']
    min_dt, safety_factor, max_newton_iter = file.meta['solver_params'].values()

    t = 0
    max_dt = t_f / 10
    # initialize outputs
    times = []
    stop_sim = False
    n_suc_steps = 0

    def get_av_interface_pos(func) -> float:
        #as_array = func.x.array
        # can maybe also do dofmap
        # brute force:
        #phi_func, _ = func.split()
        #interpolate(func, file.get_mesh())
        nx_vertices = 2 * file.params['n_x'] + 1
        ny_vertices = file.params['n_y']
        av_pos=[]
        fmesh = func.function_space().mesh()
        coords = fmesh.coordinates()
        order = np.lexsort((coords[:, 0], coords[:, 1]))
        phi = func.sub(0, deepcopy=False)
        sorted = phi.compute_vertex_values(fmesh)[order]
        phigrid = sorted.reshape(ny_vertices, nx_vertices)
        ny, nx = phigrid.shape
        interface_x = np.full(ny, np.nan)
        x_vals = np.unique(coords[:, 0])
        for i in range(ny):
            row = phigrid[i, :]
            idx = np.where(np.diff(np.sign(row)) != 0)[0] # index instance(s) of phi changing sign along row
            if len(idx) > 0:
                k = idx[0] # take first instance
                x0_interp = x_vals[k] - row[k]*(x_vals[k+1]-x_vals[k])/(row[k+1]-row[k]+1e-16) # linear interpolation
                interface_x[i] = x0_interp
        av_pos = np.nanmean(interface_x)
        print(av_pos)
        
        return av_pos

    def reset_mesh(func, file) -> mesh:
        '''
        finds the average interface position and moves
        the mesh coordinates so still logarithmic in x and linear in y
        with logarithmic spacing centered about the average
        x-position of the interface
        '''
        x0 = get_av_interface_pos(func)
        init_x_points, init_y_points = file.params['n_x'], file.params['n_y']
        Lx, Ly = file.params['Lx'], file.params['Ly']
        current_mesh = func.function_space().mesh()
        coords = current_mesh.coordinates()
        logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=init_x_points)
        logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=init_x_points)
        x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])
        # force boundaries to be the same
        x[0] = -Lx/2 
        x[-1] = Lx/2 
        for i in range(init_y_points):
            coords[i*len(x):(i+1)*len(x), 0] = x
        current_mesh.bounding_box_tree().build(current_mesh)
        #print("Min cell volume:", min(Cell(current_mesh).volume() for cell in cells(current_mesh)))
        return current_mesh

    bcs = make_bcs(file, key=boundary_key, bdval=bdval)
    V = file.make_function_space()
    v = Function(V) # this will be our solution at a time step
    dv = TrialFunction(V)
    v_prev = init_func
    v.assign(init_func)
    dphi, dpsi = TestFunctions(V)
    phi_prev, psi_prev = split(v_prev)
    phi, psi = split(v)
    dx = Measure("dx", domain=V.mesh())
    phi_save, _ = v.split(deepcopy=True)
    file.save_xdmf(phi_save, n_suc_steps)
    times.append(t)
    if track_interface:
            file.interface_positions.append(0)



    while t < t_f:
        converged = False
        attempt = 0
        while not converged:
            F_phi = ((phi - phi_prev) / dt) * dphi * dx + dot(grad(dphi), grad(psi)) * dx
            F_psi = psi * dpsi * dx - eps**2 * dot(grad(dpsi), grad(phi)) * dx - (phi**3 - phi) * dpsi * dx
            F = F_phi + F_psi
            J = derivative(F, v, dv)
            try:
                solve(F == 0, v,
                    J=J, 
                    bcs = bcs,
                    solver_parameters={'newton_solver': {'maximum_iterations': max_newton_iter}}
                    )
                converged = True
            except RuntimeError:
                dt /= 2
                print(f'Newton failed at t={t:.3e}, reducing dt to {dt:.3e}')
                if dt < min_dt:
                    print('dt below minimum threshold, simulation stopped.')
                    print(f'final time: t={t:.3e}')
                    stop_sim = True
                    break
                v.assign(v_prev) 
                attempt += 1
        if stop_sim:
            break
        # Accept the step
        n_suc_steps += 1
        t += dt
        timestring = 't = {:3.6f}\tfinal = {:3.6f}'.format(t, t_f)
        print(timestring, end='\n')
        times.append(t)
        phi_save, _ = v.split(deepcopy=True)
        file.save_xdmf(phi_save, n_suc_steps)
        v_prev.assign(v)
        x0 = get_av_interface_pos(v)
        file.interface_positions.append(x0)

        # update mesh
        if n_suc_steps % 4 == 0:
            newmesh = reset_mesh(v, file)
            V = file.make_function_space(x0=x0)
            dx = Measure("dx", domain=newmesh)
            v = Function(V)
            v_prev = interpolate(v, V)
            phi_prev, psi_prev = split(v_prev)
            dv = TrialFunction(V)
            phi, psi = split(v)
            dphi, dpsi = TestFunctions(V)
            


        # Optionally: increase dt if convergence was easy (here, always increase a bit)
        if not safe:
            dt = min(dt / safety_factor, max_dt)
                #print('warning: trying to take a step larger than courant no.')
        elif l2:
            continue
        else:
            dt = min(dt / safety_factor, max_dt)

    file.times = times
    file.meta['notes'] = notes
    file.meta['rundatetime'] = datetime.now()
    return file