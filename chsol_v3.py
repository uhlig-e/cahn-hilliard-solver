from fenics import assemble, dx, inner, grad, ln, Function, FunctionSpace, plot, RectangleMesh, Point, MixedElement, FiniteElement, conditional, lt, near, SubDomain, File, Mesh
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from dolfin import XDMFFile, MPI

#plt.style.use('my_style')
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
phi_cmap = LinearSegmentedColormap.from_list(
    "blue-green-yellow", ["blue", "green", "yellow"]
)
u_cmap = LinearSegmentedColormap.from_list(
    "black-gray-white", ["black", "gray", "white"]
)

class chsol():
    '''
    finite element solution object
    '''
    def __init__(self, filename: str, 
                 params: dict, 
                 solution_coefs=[], 
                 times=[], 
                 meta={
                     'rundatetime': None, 
                     'notes': None, 
                     'solver_params': {'min_dt': 1e-8, 'safety_factor':0.8, 'max_newton_iter': 25}
                     }
                 ):
        
        eps = params['eps']
        Lx, Ly = params['Lx'], params['Ly']
        # n_x is number of nodes per side of interface
        # n_y is number of nodes in y-direction
        n_x, n_y, deg = params['n_x'], params['n_y'], params['deg'] 
        t_f, dt = params['t_f'], params['dt']
        k = params['k']
        self.meta = meta
        self.params = params

        self.times = times
        self.filename = filename
        self.mesh_filename = filename + '_mesh.xml'
        self.make_mesh()
        self.build_xdmf()


        self.coords = None
        self.coords_order = None
        self.ic_coefs = []
        self.interface_positions = [] # redundant???

        self.eps = eps
        self.t_f = t_f
        self.k = k
        self.ell = np.sqrt(2) * eps
        self.V = 0
        self.k_amps = []
        return None
    
    def __str__(self):
        header = 'Cahn-Hilliard solve ran at:\n'
        follow = '\nwith params:\n'
        out = header + self.rundate + follow + str(self.params)
        return out  
    
    def build_xdmf(self) -> None:
        xdmf_file = XDMFFile(MPI.comm_world, f"{self.filename}.xdmf")
        xdmf_file.parameters["flush_output"] = False
        xdmf_file.parameters["functions_share_mesh"] = True
        xdmf_file.parameters["rewrite_function_mesh"] = False
        xdmf_file.close()
        return None
    
    def save_xdmf(self, function, i):
        '''
        save solution in parallel using XDMF.
        '''
        xdmf_file = XDMFFile(MPI.comm_world, f"{self.filename}.xdmf")
        xdmf_file.write_checkpoint(function, "v", i, append=True)
        xdmf_file.close()
        return None
    
    
    def make_mesh(self, x0=0, save=True) -> RectangleMesh:
        init_x_points, init_y_points = self.params['n_x'], self.params['n_y']
        Lx, Ly = self.params['Lx'], self.params['Ly']
        logmesh = RectangleMesh(Point(-Lx/2, -Ly/2), Point(Lx/2, Ly/2), 2*init_x_points, init_y_points-1) # num nodes = cells+1
        coords = logmesh.coordinates()
        logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=init_x_points)
        logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=init_x_points)
        x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])
        x[0] = -Lx/2 
        x[-1] = Lx/2 
        for i in range(init_y_points): 
            coords[i*len(x):(i+1)*len(x), 0] = x
        logmesh.bounding_box_tree().build(logmesh)
        if save: File(self.mesh_filename) << logmesh
        return logmesh  
    
    def phi_grid(self, t, return_coords=False) -> np.meshgrid:
        x0 = self.interpolate_interface_pos(t)
        mesh = self.make_mesh(x0=x0, save=False)
        nx, ny = self.params['n_x'], self.params['n_y']
        nx_vertices, ny_vertices = 2*nx + 1, ny
        coords = mesh.coordinates()
        order = np.lexsort((coords[:, 0], coords[:, 1]))  # sort by y, then x
        phi_sorted = self.phi(t).compute_vertex_values(mesh)[order]
        phigrid = phi_sorted.reshape(ny_vertices, nx_vertices) # this is the same as a meshgrid then.
        if return_coords:
            out = phigrid, coords
        else:
            out = phigrid
        return out
    
    def save(self):
        extension = '.chsol'
        file = self.filename + extension
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print('\nCH object saved to:\n' + os.getcwd() + '\n')
        return None
    
    def phi(self, t) -> Function:
        '''
        Linear interpolation of phi at time t
        '''
        x0 = self.interpolate_interface_pos(t)
        fs = self.make_function_space(x0 = x0).sub(0).collapse()
        out = Function(fs)
        xdmf = XDMFFile(f"{self.filename}.xdmf")
        if t in self.times:
            xdmf.read_checkpoint(out, 'v', self.times.index(t))
        elif t > max(self.times) or t < min(self.times):
            print('requested time outside integration limits')
        else:
            i = 0
            while self.times[i] < t:
                i += 1
                if i > len(self.times):
                    print('interpolation failed')
                    break
            diff = (t - self.times[i-1])/(self.times[i] - self.times[i-1])
            u = Function(fs)
            v = Function(fs)
            xdmf.read(u, i-1)
            xdmf.read(v, i)
            interp_coefs = u.vector().get_local() + (v.vector().get_local() - u.vector().get_local()) * diff
            out.vector().set_local(interp_coefs)
            out.vector().apply('insert')
        
        phi = out
        xdmf.close()
        return phi
    
    def make_function_space(self, x0=0) -> FunctionSpace:
        '''
        builds mesh and function space from params        
        '''
        deg = self.params['deg']
        Lx, Ly = self.params['Lx'], self.params['Ly']
        if x0 != 0:
            mesh = self.make_mesh(x0=x0, save=False)
        else:
            mesh = self.get_mesh()
        P1 = FiniteElement('P', mesh.ufl_cell(), deg) 
        # mesh.ufl_cell() changes depending upon dimension (could be interval, triangle, tetrahedron, etc.)
        element = MixedElement([P1, P1])  

        class PeriodicBoundaryY(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], Ly/2))
            
            def map(self, x, y):
                y[0] = x[0]
                y[1] = x[1] - Ly

        V = FunctionSpace(mesh, element, constrained_domain=PeriodicBoundaryY())
        return V
    
    def get_mesh(self) -> Mesh:
        out = Mesh(self.mesh_filename)
        return out

    def plot_mesh(self, save=False) -> None:
        '''
        show mesh
        '''
        mesh = self.get_mesh()
        Lx, Ly = self.params['Lx'], self.params['Ly']
        plt.figure()
        plot(mesh, color='black')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$', rotation=0, labelpad=15)
        plt.ylim([-Ly/2, Ly/2])
        plt.xlim([-Lx/2, Lx/2])
        name = self.filename + '_mesh'
        if save: plt.savefig(name, bbox_inches='tight', dpi=1200) 
        plt.show()
        return None

    def plot_profiles(self, y0=0, num_profiles=5, times=None, xbounds=(-1, 5), num_pts=600) -> None:
        x = np.linspace(xbounds[0], xbounds[1], num=num_pts)
        points = np.zeros((len(x), 2))
        points[:,0] = x
        points[:,1] = y0
        if not times:
            increment = len(self.times) // num_profiles
            times = self.times[::increment]

        for t in times: 
            phi_t = self.phi(t)
            values = np.array([phi_t(p) for p in points])
            plt.plot(x, values, label=f"t={t:.3f}")
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\phi$', rotation=0)
        plt.show()
        return None
    
    def run_validity_checks(self, save=False) -> None:
        net_phi, free_energy = [], []
        eps = self.params['eps']
        for t in self.times:
            phi = self.phi(t)
            total_phi = assemble(phi * dx)
            net_phi.append(total_phi)
            free_en = (1/4) * assemble((phi**2 - 1)**2 * dx) + (eps**2/2) * assemble(inner(grad(phi), grad(phi)) * dx)
            free_energy.append(free_en)
        net_phi, free_energy = np.array(net_phi), np.array(free_energy)
        
        fig, ax1 = plt.subplots()
        ln1 = ax1.loglog(self.times, net_phi, '-b', label=r'$\int_{\Omega} \phi \; \der \bm{x}$') 
        ax1.set_xlabel(r'$\tau$')
        ax1.set_xlim([min(self.times[1:]), max(self.times)])
        ax1.tick_params(axis='y')
        ax1.set_ylabel(r'net amount of $\phi$')
        ax2 = ax1.twinx()
        ln3 = ax2.semilogx(self.times, free_energy, '-k', label=r'$\mathcal{F}$')
        ax2.set_ylabel(r'free energy, $\mathcal{F}$')
        ax2.tick_params(axis='y')
        # Combine legends
        lns = ln1 + ln3 
        labels = [l.get_label() for l in lns]
        ax2.legend(lns, labels, loc='best')
        if not save:
            plt.show()
        else:
            picname = self.filename + 'NCs'
            plt.savefig(picname, bbox_inches='tight')

        return None
    
    def phi_zero_interface(self, t):
        '''
        searches along each y-row for a change of sign in phi, 
        then determines the x-position of the crossing
        note that the x-value thing may not work for more unstructured grids...
        '''
        grid = self.phi_grid(t)
        ny, nx = grid.shape
        interface_x = np.full(ny, np.nan)
        x0 = self.interpolate_interface_pos(t)
        x_vals = np.unique(self.make_mesh(x0=x0, save=False).coordinates()[:, 0])
        for i in range(ny):
            row = grid[i, :]
            idx = np.where(np.diff(np.sign(row)) != 0)[0] # index instance(s) of phi changing sign along row
            if len(idx) > 0:
                k = idx[0] # take first instance of crossing zero
                x0_interp = x_vals[k] - row[k]*(x_vals[k+1]-x_vals[k])/(row[k+1]-row[k]+1e-16) # linear interpolation
                interface_x[i] = x0_interp
        return interface_x
    
    def get_av_interface_displacements(self):
        if not self.interface_positions:
            positions = np.array([np.nanmean(self.phi_zero_interface(t)) for t in self.times])
            positions -= positions[0]
            positions = np.abs(positions)
            self.interface_positions = positions
            self.save()
        else:
            position = self.interface_positions
        return positions
    
    def get_V(self):
        t = np.array(self.times)**(1/2)
        if not self.interface_positions:
            self.V = np.nanmean(self.get_av_interface_displacements() / 2 / t)
        else:
            self.V = np.nanmean(self.interface_positions / 2 / t)
        self.save()
        return None
    
    def plot_mode_amp(self, diff_times=None, show=False):
        if diff_times:
            times = diff_times
        else:
            times = self.times

        most_dangerous_wavenumber_amps = []

        if not diff_times and self.k_amps:
            most_dangerous_wavenumber_amps = self.k_amps
        else:
            for t in times:
                interfacex = self.phi_zero_interface(t)
                mask = ~np.isnan(interfacex)
                h = interfacex[mask] - np.nanmean(interfacex)
                fth = np.fft.fft(h)
                amplitude_spectrum = np.abs(fth) / len(h)
                positive_amplitudes = amplitude_spectrum[:len(h) // 2] * 2 # Multiply by 2 for the single-sided amplitude
                most_dangerous_wavenumber_amps.append(positive_amplitudes[np.argmax(positive_amplitudes)]) # find a way to save which wavenumber is dominant at each timestep!!!
                # also find a way to interpolate these!!!!

        if not diff_times and not self.k_amps: 
            self.k_amps = most_dangerous_wavenumber_amps
            self.save()

        if show:
            plt.figure()
            plt.loglog(times, most_dangerous_wavenumber_amps, 'ok')
            plt.xlabel(r'time, $t$')
            plt.ylabel(r'seeded mode amplitude')
            #plt.savefig('seeded mode growth', dpi=1200, bbox_inches='tight')
            plt.show()
        return None
    
    def plot_spectrum(self, times=None, num_spectra=5):
        if not times:
            times = self.times[::len(self.times)//num_spectra]
        for t in times:
            interfacex = self.phi_zero_interface(t)
            mask = ~np.isnan(interfacex)
            h = interfacex[mask] - np.nanmean(interfacex)
            fth = np.fft.fft(h)
            amplitude_spectrum = np.abs(fth) / len(h)
            frequency_axis = np.fft.fftfreq(len(h), d=1)
            positive_frequencies = frequency_axis[:len(h) // 2]
            positive_amplitudes = amplitude_spectrum[:len(h) // 2] * 2 # Multiply by 2 for the single-sided amplitude
            most_dangerous_wavenumber_amp = positive_amplitudes[np.argmax(positive_amplitudes)]
        plt.figure()
        plt.plot(positive_frequencies, positive_amplitudes)        
        plt.xlabel(r'$2\pi k$ [wavenumber]')
        plt.ylabel(r'$|a|$ [amplitude norm]')
        plt.show()

        return None
    
    def interpolate_interface_pos(self, t):
        if t > max(self.times):
            print('Error: t={} is outside of solved times (cannot extrapolate)'.format(t))
        elif min(self.times) < t < max(self.times):
            j = 0
            while j < len(self.times)-1:
                if self.times[j] < t < self.times[j+1]:
                    break
                else:
                    j += 1
            interp_result = self.interface_positions[j] + (self.interface_positions[j+1] - self.interface_positions[j])/(self.times[j+1] - self.times[j]) * (t-self.times[j])
        else:
            print('Error...negative time?')
        
        return interp_result
                 
    def plot_dendrite(self, num=5) -> None:
        '''
        needs:
        interface positions
        mode amplitudes
        '''
        increment = len(self.times) // num
        times = self.times[::increment]
        shifts = self.interface_positions[::increment]
        fig, ax = plt.subplots(constrained_layout=True)
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        n_x, n_y = self.params['n_x'], self.params['n_y']
        x0=0
        logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=n_x)
        logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=n_x)
        x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])
        y=np.linspace(-Ly/2, Ly/2, num=n_y)
        x[0] = -Lx/2 
        x[-1] = Lx/2
        handles = []
        for i, t in enumerate(times):
            X, Y = np.meshgrid(x - shifts[i], y) # need to interpolate shifts
            cs = ax.contour(
                X, Y, self.phi_grid(t),
                levels=[0],
                linewidths=1.5,
                origin='lower',
            )
            line = plt.Line2D([0],[0])
            handles.append(line)
        ax.legend(handles, [f"t={t:.2f}" for t in times])
        #plt.xlim([0, 2]) 
        plt.xlim([0, 1.1*max(self.k_amps)])
        plt.ylim([0, int(1/self.k/2)])
        plt.show()
        return None
    
    def make_movie(self, new_times=None, vmin=None, vmax=None,
               show_colorbar=True, custom_cmap=False,
               skip=1, bounds=None,
               interface_positions=None,
               single_wavelength=False):

        if custom_cmap:
            cmap = custom_cmap
        else:
            cmap = phi_cmap

        Lx, Ly = self.params['Lx'], self.params['Ly']
        times = self.times
        if new_times:
            times = new_times

        # Get coordinates from first frame
        x_vals, y_vals = self.mesh_coordinates(times[0])

        if bounds:
            lx, ly = bounds
            ax_xlim = (-lx/2, lx/2)
            ax_ylim = (-ly/2, ly/2)
        else:
            ax_xlim = (-Lx/2, Lx/2)
            ax_ylim = (-Ly/2, Ly/2)

        if single_wavelength:
            ax_ylim = (0, single_wavelength)

        fig, ax = plt.subplots(constrained_layout=True)

        initial_frame = self.phi_grid(times[0])

        # ---- CREATE PCOLORMESH ----
        quad = ax.pcolormesh(
            x_vals,
            y_vals,
            initial_frame,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto"
        )

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$', rotation=0, labelpad=15)

        time_title = ax.set_title(r'$t$ = 0.000')

        contour_holder = [None]

        contour_holder[0] = ax.contour(
            x_vals,
            y_vals,
            initial_frame,
            levels=[0],
            colors='black',
            linewidths=1
        )

        if show_colorbar:
            cbar = plt.colorbar(quad, ax=ax, fraction=0.05, pad=0.04)
            cbar.set_label(r'$\phi$', labelpad=10, rotation=0)

        def update(indx):

            actual_index = indx * skip
            frame = self.phi_grid(times[actual_index])
            x_vals, y_vals = self.mesh_coordinates(times[actual_index])

            # Update pcolormesh values
            quad.set_array(frame.ravel())

            time_title.set_text(rf'$t$ = {times[actual_index]:1.3f}')

            # Remove old contour
            if contour_holder[0] is not None:
                for coll in contour_holder[0].collections:
                    coll.remove()

            # Handle shifting window if requested
            if interface_positions is not None:
                shift = interface_positions[actual_index]
                ax.set_xlim(ax_xlim[0] + shift, ax_xlim[1] + shift)

            contour_holder[0] = ax.contour(
                x_vals,
                y_vals,
                frame,
                levels=[0],
                colors='black',
                linewidths=2
            )

            return [quad]

        movie_filename = self.filename + '.mp4'

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(times[::skip]),
            blit=False,   # IMPORTANT: blitting does not work well with pcolormesh
            interval=50
        )

        ani.save(movie_filename, writer='ffmpeg', fps=10)
        plt.close(fig)

        return None