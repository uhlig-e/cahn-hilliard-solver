from fenics import assemble, dx, inner, grad, ln, Function, FunctionSpace, plot, RectangleMesh, Point, MixedElement, FiniteElement, conditional, lt, near, SubDomain, File, Mesh
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from dolfin import XDMFFile, MPI
import json

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
        n_x, n_y, deg = params['n_x'], params['n_y'], params['deg'] 
        t_f, dt = params['t_f'], params['dt']
        k = params['k']
        self.meta = meta
        self.params = params

        self.times = times
        self.filename = filename
        self.mesh_filename = filename + '_mesh.xml'
        self.mesh_trajectory_dir = filename + '_mesh_trajectory'
        self.solution_data_file = filename + '_solutions.h5'
        self.mesh_metadata_file = filename + '_mesh_metadata.json'
        
        # Create directory for mesh trajectory if it doesn't exist
        os.makedirs(self.mesh_trajectory_dir, exist_ok=True)
        
        self.make_mesh()
        self.build_xdmf()

        self.coords = None
        self.coords_order = None
        self.ic_coefs = []
        self.interface_positions = []
        self.mesh_metadata = {}  # Track mesh state at each timestep - uses int keys

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
    
    def save_solution_with_mesh(self, function, mesh, timestep, t):
        '''
        Save solution function and its corresponding mesh at each timestep.
        Stores mesh in separate directory and solution data in HDF5 format.
        '''
        # Save mesh to individual file
        mesh_file = os.path.join(self.mesh_trajectory_dir, f'mesh_{timestep:06d}.xml')
        File(mesh_file) << mesh
        
        # Save function to XDMF (as before, for compatibility)
        xdmf_file = XDMFFile(MPI.comm_world, f"{self.filename}.xdmf")
        xdmf_file.write_checkpoint(function, "v", timestep, append=True)
        xdmf_file.close()
        
        # Store metadata about this timestep's mesh
        # Use integer key to avoid JSON string conversion issues
        self.mesh_metadata[int(timestep)] = {
            'time': float(t),
            'mesh_file': mesh_file,
            'num_vertices': mesh.num_vertices(),
            'num_cells': mesh.num_cells()
        }
        print(f"Saved timestep {timestep} to {mesh_file}")
        return None
    
    def save_xdmf(self, function, i):
        '''
        save solution in parallel using XDMF.
        (Legacy method - now use save_solution_with_mesh)
        '''
        xdmf_file = XDMFFile(MPI.comm_world, f"{self.filename}.xdmf")
        xdmf_file.write_checkpoint(function, "v", i, append=True)
        xdmf_file.close()
        return None
    
    def save_mesh_metadata(self):
        '''
        Save mesh metadata to JSON file for later reference
        '''
        # Convert integer keys to strings for JSON compatibility
        metadata_to_save = {str(k): v for k, v in self.mesh_metadata.items()}
        with open(self.mesh_metadata_file, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        print(f"Saved mesh metadata to {self.mesh_metadata_file}")
        print(f"Metadata keys: {list(self.mesh_metadata.keys())}")
        return None
    
    def load_mesh_metadata(self):
        '''
        Load mesh metadata from JSON file and convert back to integer keys
        '''
        if os.path.exists(self.mesh_metadata_file):
            with open(self.mesh_metadata_file, 'r') as f:
                metadata_from_file = json.load(f)
            # Convert string keys back to integers
            self.mesh_metadata = {int(k): v for k, v in metadata_from_file.items()}
            print(f"Loaded mesh metadata with keys: {list(self.mesh_metadata.keys())}")
        else:
            print(f"Warning: Mesh metadata file not found at {self.mesh_metadata_file}")
        return None
    
    def get_mesh_at_timestep(self, timestep):
        '''
        Load the specific mesh that was used at a given timestep
        '''
        timestep = int(timestep)
        
        if not self.mesh_metadata:
            self.load_mesh_metadata()
        
        if timestep in self.mesh_metadata:
            mesh_file = self.mesh_metadata[timestep]['mesh_file']
            if os.path.exists(mesh_file):
                print(f"Loading mesh from {mesh_file}")
                return Mesh(mesh_file)
            else:
                print(f"Warning: Mesh file not found at {mesh_file}")
        else:
            print(f"Warning: Timestep {timestep} not in metadata. Available: {list(self.mesh_metadata.keys())}")
        
        # Fallback: use initial mesh
        print(f"Using initial mesh as fallback")
        return self.get_mesh()
    
    def make_mesh(self, x0=0, save=True) -> RectangleMesh:
        init_x_points, init_y_points = self.params['n_x'], self.params['n_y']
        Lx, Ly = self.params['Lx'], self.params['Ly']
        logmesh = RectangleMesh(Point(-Lx/2, -Ly/2), Point(Lx/2, Ly/2), 2*init_x_points, init_y_points-1)
        coords = logmesh.coordinates()
        logarrayright = np.logspace(-6, np.log10(Lx/2 - x0), num=init_x_points)
        logarrayleft = np.logspace(-6, np.log10(np.abs(-Lx/2 - x0)), num=init_x_points)
        x = np.hstack([x0-logarrayleft[::-1], [x0], x0+logarrayright])
        x[0] = -Lx/2 
        x[-1] = Lx/2 
        for i in range(init_y_points): 
            coords[i*len(x):(i+1)*len(x), 0] = x
        logmesh.bounding_box_tree().build(logmesh)
        if save: 
            File(self.mesh_filename) << logmesh
        return logmesh  
    
    def phi_grid(self, t, return_coords=False) -> np.meshgrid:
        '''
        Get phi as a grid at time t, using the appropriate mesh for that time
        '''
        # Find closest timestep
        if len(self.times) == 0:
            print("Warning: No times available")
            return np.array([])
        
        idx = min(range(len(self.times)), key=lambda i: abs(self.times[i] - t))
        
        phi = self.phi(t)
        mesh = self.get_mesh_at_timestep(idx)
        
        nx, ny = self.params['n_x'], self.params['n_y']
        nx_vertices, ny_vertices = 2*nx + 1, ny
        coords = mesh.coordinates()
        order = np.lexsort((coords[:, 0], coords[:, 1]))
        phi_sorted = phi.compute_vertex_values(mesh)[order]
        phigrid = phi_sorted.reshape(ny_vertices, nx_vertices)
        
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
        # Also save mesh metadata
        self.save_mesh_metadata()
        print('\nCH object saved to:\n' + os.getcwd() + '\n')
        return None
    
    def phi(self, t) -> Function:
        '''
        Linear interpolation of phi at time t, using the mesh appropriate for that timestep
        '''
        if len(self.times) == 0:
            print("Error: No solution times available")
            return None
        
        # Find which timestep this falls into
        if t in self.times:
            idx = self.times.index(t)
            mesh = self.get_mesh_at_timestep(idx)
        elif t > max(self.times) or t < min(self.times):
            print(f'requested time {t} outside integration limits [{min(self.times)}, {max(self.times)}]')
            mesh = self.get_mesh()
        else:
            # Find interpolation interval
            i = 0
            while i < len(self.times)-1 and self.times[i] < t:
                i += 1
            
            # Use the mesh from the earlier timestep for interpolation
            mesh = self.get_mesh_at_timestep(i-1)
        
        fs = self.make_function_space(mesh=mesh)
        out = Function(fs)
        xdmf = XDMFFile(f"{self.filename}.xdmf")
        
        if t in self.times:
            idx = self.times.index(t)
            xdmf.read_checkpoint(out, 'v', idx)
        elif t > max(self.times) or t < min(self.times):
            print('requested time outside integration limits')
        else:
            i = 0
            while i < len(self.times)-1 and self.times[i] < t:
                i += 1
            
            if i > len(self.times) - 1:
                print('interpolation failed')
                xdmf.close()
                return out
            
            diff = (t - self.times[i-1])/(self.times[i] - self.times[i-1])
            u = Function(fs)
            v = Function(fs)
            xdmf.read_checkpoint(u, 'v', i-1)
            xdmf.read_checkpoint(v, 'v', i)
            interp_coefs = u.vector().get_local() + (v.vector().get_local() - u.vector().get_local()) * diff
            out.vector().set_local(interp_coefs)
            out.vector().apply('insert')
        
        xdmf.close()
        return out
    
    def make_function_space(self, x0=0, mesh=None) -> FunctionSpace:
        '''
        builds mesh and function space from params        
        '''
        deg = self.params['deg']
        Lx, Ly = self.params['Lx'], self.params['Ly']
        if x0 == 0 and not mesh:
            mesh = self.get_mesh()
        elif x0 != 0 and not mesh:
            mesh = self.make_mesh(x0=x0, save=False)
        elif mesh and x0 != 0:
            print('x0 method of mesh generation deprecated. using mesh provided to make_function_space')

        P1 = FiniteElement('P', mesh.ufl_cell(), deg) 
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
        for idx, t in enumerate(self.times):
            phi = self.phi(t)
            mesh = self.get_mesh_at_timestep(idx)
            total_phi = assemble(phi * dx(domain=mesh))
            net_phi.append(total_phi)
            free_en = (1/4) * assemble((phi**2 - 1)**2 * dx(domain=mesh)) + (eps**2/2) * assemble(inner(grad(phi), grad(phi)) * dx(domain=mesh))
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
        '''
        grid = self.phi_grid(t)
        
        if grid.size == 0:
            return np.array([])
        
        ny, nx = grid.shape
        interface_x = np.full(ny, np.nan)
        
        # Get the mesh at this timestep to get x_vals
        if len(self.times) == 0:
            return interface_x
        
        idx = min(range(len(self.times)), key=lambda i: abs(self.times[i] - t))
        mesh = self.get_mesh_at_timestep(idx)
        x_vals = np.unique(mesh.coordinates()[:, 0])
        
        for i in range(ny):
            row = grid[i, :]
            idx_cross = np.where(np.diff(np.sign(row)) != 0)[0]
            if len(idx_cross) > 0:
                k = idx_cross[0]
                x0_interp = x_vals[k] - row[k]*(x_vals[k+1]-x_vals[k])/(row[k+1]-row[k]+1e-16)
                interface_x[i] = x0_interp
        return interface_x
    
    def get_av_interface_displacements(self):
        '''
        Compute average interface position at each timestep from saved solutions.
        (Always recomputed for accuracy - not stored during solve)
        '''
        print("Computing interface positions from saved solutions...")
        positions = []
        for i, t in enumerate(self.times):
            if i % max(1, len(self.times)//10) == 0:
                print(f"  Timestep {i}/{len(self.times)}", end='\r')
            
            interface_row = self.phi_zero_interface(t)
            avg_pos = np.nanmean(interface_row)
            positions.append(avg_pos)
        
        positions = np.array(positions)
        print(f"\nInterface positions computed")
        print(f"  Initial: {positions[0]:.6f}")
        print(f"  Final:   {positions[-1]:.6f}")
        
        self.interface_positions = positions
        self.save()
        
        return self.interface_positions
    
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
                print(interfacex)
                mask = ~np.isnan(interfacex)
                if len(interfacex[mask]) > 0:
                    # Interpolate to uniform grid for consistent FFT array length
                    y_indices = np.arange(len(interfacex))
                    h_uniform = np.interp(y_indices, y_indices[mask], interfacex[mask])
                    h = h_uniform - np.mean(h_uniform)
                    fth = np.fft.fft(h)
                    amplitude_spectrum = np.abs(fth) / len(h)
                    positive_amplitudes = amplitude_spectrum[:len(h) // 2] * 2
                    most_dangerous_wavenumber_amps.append(positive_amplitudes[np.argmax(positive_amplitudes)])
                else:
                    most_dangerous_wavenumber_amps.append(0)

        if not diff_times and not self.k_amps: 
            self.k_amps = most_dangerous_wavenumber_amps
            self.save()

        if show:
            c_infty = 0.05
            a_0 = 0.05
            theor = lambda t: c_infty/np.sqrt(8 * np.pi * t) * self.k - np.sqrt(2)/3 * self.eps * self.k**3
            plt.figure()
            plt.loglog(times, most_dangerous_wavenumber_amps, 'ok')
            plt.loglog(times, [a_0 * np.exp(theor(t) * t) for t in times], '*r')
            plt.xlabel(r'time, $t$')
            plt.ylabel(r'seeded mode amplitude')
            plt.show()
        return None
    
    def plot_spectrum(self, times=None, num_spectra=5):
        if not times:
            times = self.times[::len(self.times)//num_spectra]
        for t in times:
            interfacex = self.phi_zero_interface(t)
            mask = ~np.isnan(interfacex)
            if len(interfacex[mask]) > 0:
                h = interfacex[mask] - np.nanmean(interfacex)
                fth = np.fft.fft(h)
                amplitude_spectrum = np.abs(fth) / len(h)
                frequency_axis = np.fft.fftfreq(len(h), d=1)
                positive_frequencies = frequency_axis[:len(h) // 2]
                positive_amplitudes = amplitude_spectrum[:len(h) // 2] * 2
                
                plt.figure()
                plt.plot(positive_frequencies, positive_amplitudes)        
                plt.xlabel(r'$2\pi k$ [wavenumber]')
                plt.ylabel(r'$|a|$ [amplitude norm]')
                plt.show()

        return None
    
    def interpolate_interface_pos(self, t):
        if len(self.interface_positions) == 0:
            return 0
        
        if t > max(self.times):
            print('Error: t={} is outside of solved times (cannot extrapolate)'.format(t))
            return self.interface_positions[-1]
        elif min(self.times) < t < max(self.times):
            j = 0
            while j < len(self.times)-1:
                if self.times[j] < t < self.times[j+1]:
                    break
                else:
                    j += 1
            interp_result = self.interface_positions[j] + (self.interface_positions[j+1] - self.interface_positions[j])/(self.times[j+1] - self.times[j]) * (t-self.times[j])
        else:
            print('Error: negative time?')
            interp_result = self.interface_positions[0]
        
        return interp_result
    
    def mesh_coordinates(self, t):
        '''
        Get mesh coordinates at time t, returning sorted unique x and y values
        '''
        if len(self.times) == 0:
            mesh = self.get_mesh()
        else:
            idx = min(range(len(self.times)), key=lambda i: abs(self.times[i] - t))
            mesh = self.get_mesh_at_timestep(idx)
        coords = mesh.coordinates()
        x_vals = np.unique(coords[:, 0])
        y_vals = np.unique(coords[:, 1])
        return x_vals, y_vals

    def plot_dendrite(self, num=5, show_region=None) -> None:
        '''
        Plot dendrite morphology with interface positioned at x=0 for each frame.
        Each frame shows phi=0 interface centered at x=0.
        
        Parameters:
        -----------
        num : int
            Number of frames to show
        show_region : tuple
            (x_min, x_max, y_min, y_max) to specify viewing region, or None for auto
        '''
        if len(self.times) == 0:
            print("No solution data available")
            return None
        
        # Ensure interface positions are computed
        if len(self.interface_positions) == 0:
            print("Computing interface positions...")
            self.get_av_interface_displacements()
        
        increment = max(1, len(self.times) // num)
        times_to_plot = self.times[::increment]
        
        fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 8))
        Ly = self.params['Ly']
        
        handles = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
        
        for plot_idx, t in enumerate(times_to_plot):
            # Get phi grid and mesh at this timestep
            phi_grid = self.phi_grid(t)
            x_mesh, y_mesh = self.mesh_coordinates(t)
            
            # Compute the interface position at this timestep
            interface_row = self.phi_zero_interface(t)
            interface_x_computed = np.nanmean(interface_row)
            
            # Create meshgrid from actual mesh coordinates
            X, Y = np.meshgrid(x_mesh, y_mesh)
            
            # Shift x-coordinates so interface is at x=0
            X_shifted = X - interface_x_computed
            
            # Draw contour at phi=0
            cs = ax.contour(
                X_shifted, Y, phi_grid,
                levels=[0],
                colors=[colors[plot_idx]],
                linewidths=2,
                alpha=0.8
            )
            
            # Create legend handle
            line = plt.Line2D([0],[0], color=colors[plot_idx], linewidth=2)
            handles.append(line)
        
        ax.legend(handles, [f"t={t:.3f}" for t in times_to_plot], loc='best')
        
        # Set axis limits
        if show_region is not None:
            x_min, x_max, y_min, y_max = show_region
        else:
            # Auto limits: show region from interface (x=0) to right boundary
            x_min = 0
            if self.k_amps:
                x_max = 1.1 * max(self.k_amps)
            else:
                x_max = 1.0
            y_min = 0
            y_max = 1/self.k/2
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Interface (x=0)')
        ax.set_xlabel(r'$x - x_{\text{interface}}$', fontsize=12)
        ax.set_ylabel(r'$y$', fontsize=12, rotation=0, labelpad=15)
        ax.set_title(r'Dendrite Growth (Interface Centered at x=0)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.show()
        return None

    def make_movie(self, new_times=None, vmin=None, vmax=None,
                show_colorbar=True, custom_cmap=False,
                skip=1, bounds=None,
                interface_positions=None,
                single_wavelength=False):
        '''
        Create movie of phi field evolution with interface contours
        
        Parameters:
        -----------
        new_times : list, optional
            Custom times to use instead of self.times
        vmin, vmax : float, optional
            Color scale limits
        show_colorbar : bool
            Whether to show colorbar
        custom_cmap : colormap, optional
            Custom colormap
        skip : int
            Frame skip (default 1 = show every frame)
        bounds : tuple, optional
            (Lx, Ly) to set custom domain bounds
        interface_positions : array, optional
            Interface positions for shifting window
        single_wavelength : float, optional
            Show only single wavelength in y-direction
        '''
        
        if len(self.times) == 0:
            print("No solution data available")
            return None
        
        if custom_cmap:
            cmap = custom_cmap
        else:
            cmap = phi_cmap

        Lx, Ly = self.params['Lx'], self.params['Ly']
        times = self.times if new_times is None else new_times

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

        # Store contour object
        contour_holder = [None]

        # Create initial contour
        contour_holder[0] = ax.contour(
            x_vals,
            y_vals,
            initial_frame,
            levels=[0],
            colors='black',
            linewidths=1.5
        )

        if show_colorbar:
            cbar = plt.colorbar(quad, ax=ax, fraction=0.05, pad=0.04)
            cbar.set_label(r'$\phi$', labelpad=10, rotation=0)

        def update(frame_num):
            '''
            Update function for animation
            '''
            actual_index = frame_num * skip
            
            if actual_index >= len(times):
                actual_index = len(times) - 1
            
            # Get frame data
            frame = self.phi_grid(times[actual_index])
            x_vals_frame, y_vals_frame = self.mesh_coordinates(times[actual_index])

            # Update pcolormesh values
            quad.set_array(frame.ravel())

            time_title.set_text(rf'$t$ = {times[actual_index]:.3f}')

            # Remove old contour lines
            if contour_holder[0] is not None:
                for coll in contour_holder[0].collections:
                    coll.remove()

            # Handle shifting window if requested
            if interface_positions is not None:
                shift = interface_positions[actual_index]
                ax.set_xlim(ax_xlim[0] + shift, ax_xlim[1] + shift)

            # Draw new contour
            contour_holder[0] = ax.contour(
                x_vals_frame,
                y_vals_frame,
                frame,
                levels=[0],
                colors='black',
                linewidths=1.5
            )

            return [quad]

        movie_filename = self.filename + '.mp4'
        
        print(f"Creating movie: {movie_filename}")
        print(f"Number of frames: {len(times[::skip])}")
        print(f"Time range: {times[0]:.3e} to {times[-1]:.3e}")

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(times[::skip]),
            blit=False,
            interval=100,
            repeat=True
        )

        # Save with ffmpeg
        try:
            print("Attempting to save with ffmpeg...")
            ani.save(movie_filename, writer='ffmpeg', fps=10, dpi=80)
            print(f"Movie saved successfully to {movie_filename}")
        except Exception as e:
            print(f"Error saving movie with ffmpeg: {e}")
            print("Attempting to save with pillow writer instead...")
            try:
                gif_filename = movie_filename.replace('.mp4', '.gif')
                ani.save(gif_filename, writer='pillow', fps=10)
                print(f"Saved as GIF: {gif_filename}")
            except Exception as e2:
                print(f"Error saving with pillow: {e2}")
                print("Movie creation failed - check ffmpeg installation")

        plt.close(fig)
        return None
    
    def diagnose_interface_discrepancy(self, timestep_idx):
        '''
        Compare interface detection methods for a specific timestep
        to identify the source of discrepancies.
        '''
        t = self.times[timestep_idx]
        
        print(f"\n{'='*60}")
        print(f"Diagnosing interface detection at timestep {timestep_idx}, t={t:.6f}")
        print(f"{'='*60}")
        
        # Method 1: Get phi via phi() method (post-processing path)
        phi_func = self.phi(t)
        mesh_loaded = self.get_mesh_at_timestep(timestep_idx)
        
        # Method 2: Get phi_grid (what plot_dendrite uses)
        phi_grid = self.phi_grid(t)
        
        # Compute interface for phi_grid method
        if phi_grid.size > 0:
            ny, nx = phi_grid.shape
            interface_x_grid = np.full(ny, np.nan)
            x_mesh, y_mesh = self.mesh_coordinates(t)
            x_vals = np.unique(x_mesh)
            
            for i in range(ny):
                row = phi_grid[i, :]
                idx = np.where(np.diff(np.sign(row)) != 0)[0]
                if len(idx) > 0:
                    k = idx[0]
                    x_interp = x_vals[k] - row[k]*(x_vals[k+1]-x_vals[k])/(row[k+1]-row[k]+1e-16)
                    interface_x_grid[i] = x_interp
            
            av_grid = np.nanmean(interface_x_grid)
        else:
            av_grid = np.nan
        
        # Compare with stored value
        stored = self.interface_positions[timestep_idx]
        
        print(f"Stored interface position:         {stored:.10f}")
        print(f"Freshly computed (phi_grid):       {av_grid:.10f}")
        print(f"Absolute difference:               {abs(stored - av_grid):.2e}")
        print(f"Relative difference:               {abs(stored - av_grid)/abs(stored)*100:.4f}%")
        
        # Check mesh properties
        print(f"\nMesh properties:")
        print(f"  Loaded mesh vertices:  {mesh_loaded.num_vertices()}")
        print(f"  Loaded mesh cells:     {mesh_loaded.num_cells()}")
        coords = mesh_loaded.coordinates()
        print(f"  X coordinate range:    [{coords[:,0].min():.6f}, {coords[:,0].max():.6f}]")
        print(f"  Y coordinate range:    [{coords[:,1].min():.6f}, {coords[:,1].max():.6f}]")
        
        # Check function space
        print(f"\nFunction space properties:")
        print(f"  Function dofs:         {phi_func.function_space().dim()}")
        print(f"  Function min value:    {phi_func.vector().min():.6f}")
        print(f"  Function max value:    {phi_func.vector().max():.6f}")
        
        print(f"{'='*60}\n")
        
        return stored, av_grid
