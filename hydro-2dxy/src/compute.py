# src/compute.py
import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
#  from .tools.init import save_3D_plot

def discrete_H( h: np.ndarray, eta: np.ndarray) -> tuple:
    """
    Equations (47), (48), and (49): to compute water depth at cell centers and staggered grid points.

    Args:
        h: Bottom topography at cell centers 
        eta: Surface elevation at cell centers 

    Returns:
        Hc:  Total water depth at cell centers  (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
    """
    Ny, Nx = h.shape
    edge = 1e4 

    # Total water depth at cell centers
    Hc = np.clip(h + eta, h, edge)

    # Total water depth at u-grid points (i+1/2, j)
    h_u   = np.zeros((Ny, Nx + 1))
    eta_u = np.zeros((Ny, Nx + 1))

    h_u[:, 1:Nx] = 0.5 * (h[:, 0:Nx - 1] + h[:, 1:Nx])
    h_u[:, 0]    = h[:, 0]
    h_u[:, Nx]   = h[:, Nx - 1]

    eta_u[:, 1:Nx] = 0.5 * (eta[:, 0:Nx - 1] + eta[:, 1:Nx])
    eta_u[:, 0]    = eta[:, 0]
    eta_u[:, Nx]   = eta[:, Nx - 1]
    H_u = np.clip(h_u + eta_u, h_u, edge)

    # Total water depth at v-grid points (i, j+1/2)
    h_v   = np.zeros((Ny + 1, Nx))
    eta_v = np.zeros((Ny + 1, Nx))

    h_v[1:Ny, :] = 0.5 * (h[0:Ny - 1, :] + h[1:Ny, :])
    h_v[0, :]    = h[0, :]
    h_v[Ny, :]   = h[Ny - 1, :]

    eta_v[1:Ny, :] = 0.5 * (eta[0:Ny - 1, :] + eta[1:Ny, :])
    eta_v[0, :]    = eta[0, :]
    eta_v[Ny, :]   = eta[Ny - 1, :]

    H_v = np.clip(h_v + eta_v, h_v, edge)

    return Hc, H_u, H_v

def matrix_F(config,u: np.ndarray, v: np.ndarray) -> np.ndarray: 
    """
    New
    Compute the Finite Difference Operator F for both u and v (Equations 50.1 and 51.1).
    """
    Ny, Nx_u = u.shape
    Ny_v, Nx = v.shape
    dx = config.grid.dx
    dy = config.grid.dy
    nu = config.physical.nu
    dt = config.time.dt
    Fu = np.zeros_like(u)
    Fv = np.zeros_like(v)

    # Compute Fu at vertical faces (i+1/2, j)
    for j in range(1, Ny - 1):
        for i in range(1, Nx_u - 1):
            # Advective term using upwind scheme
            adv_u = u[j, i] * (u[j, i] - u[j, i - 1]) / dx + \
                    v[j, i] * (u[j, i] - u[j - 1, i]) / dy

            # Viscous term using central differences
            visc_u = nu * ((u[j, i + 1] - 2 * u[j, i] + u[j, i - 1]) / dx ** 2 +
                           (u[j + 1, i] - 2 * u[j, i] + u[j - 1, i]) / dy ** 2)

            Fu[j, i] = u[j, i] + dt * (-adv_u + visc_u)

    # Compute Fv at horizontal faces (i, j+1/2)
    for j in range(1, Ny_v - 1):
        for i in range(1, Nx - 1):
            adv_v = u[j, i] * (v[j, i] - v[j, i - 1]) / dx + \
                    v[j, i] * (v[j, i] - v[j - 1, i]) / dy

            visc_v = nu * ((v[j, i + 1] - 2 * v[j, i] + v[j, i - 1]) / dx ** 2 +
                           (v[j + 1, i] - 2 * v[j, i] + v[j - 1, i]) / dy ** 2)

            Fv[j, i] = v[j, i] + dt * (-adv_v + visc_v)

    return Fu, Fv

def matrix_G(config,H_u: np.ndarray, H_v: np.ndarray, Fu: np.ndarray, Fv: np.ndarray) -> tuple:
    """
    Compute the flux terms Gu and Gv.

    Parameters:
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        Fu: Explicit term for u-velocity (Ny, Nx + 1)
        Fv: Explicit term for v-velocity (Ny + 1, Nx)
        gamma_T: Wind stress coefficient
        u_a: Wind speed in x-direction
        v_a: Wind speed in y-direction
        dt: Time step

    Returns:
        Gu: Flux term for u-velocity (Ny, Nx + 1)
        Gv: Flux term for v-velocity (Ny + 1, Nx)
    """
    dt = config.time.dt
    gamma_T = config.physical.gamma_T
    u_a = config.physical.u_a
    v_a = config.physical.v_a

    Gu = H_u * Fu + dt * gamma_T * u_a
    Gv = H_v * Fv + dt * gamma_T * v_a

    return Gu, Gv

def matrix_T(config, Hc: np.ndarray) -> csr_matrix:
    """
    Assemble the matrix T for solving eta_new.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        T: Sparse matrix T (N, N) where N = Ny * Nx
    """
    gamma = config.physical.gamma
    
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    g = config.physical.g
    Ny, Nx = Hc.shape
    N = Ny * Nx
    T = lil_matrix((N, N))
    gamma = np.ones((Ny, Nx)) * gamma
    gamma_dt = gamma * dt
    dt2_dx2 = (dt ** 2) / (dx ** 2)
    dt2_dy2 = (dt ** 2) / (dy ** 2)

    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            H_val = Hc[j, i]
            denom = H_val + gamma_dt[j, i] if H_val + gamma_dt[j, i] != 0 else 1e-6
            coeff = (H_val ** 2) / denom

            diagonal = H_val + g * coeff * (dt2_dx2 + dt2_dy2)
            T[idx, idx] = diagonal

            # Off-diagonal elements
            if i > 0:
                idx_left = idx - 1
                T[idx, idx_left] = -g * coeff * dt2_dx2
            if i < Nx - 1:
                idx_right = idx + 1
                T[idx, idx_right] = -g * coeff * dt2_dx2
            if j > 0:
                idx_down = idx - Nx
                T[idx, idx_down] = -g * coeff * dt2_dy2
            if j < Ny - 1:
                idx_up = idx + Nx
                T[idx, idx_up] = -g * coeff * dt2_dy2

    return T.tocsr()

def vector_b(config, Hc: np.ndarray, H_u: np.ndarray, H_v: np.ndarray, Gu: np.ndarray, 
             Gv: np.ndarray) -> np.ndarray:
    """
    Compute the right-hand side vector b for solving eta_new.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        Gu: Flux term for u-velocity (Ny, Nx + 1)
        Gv: Flux term for v-velocity (Ny + 1, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        b: Right-hand side vector b (N,)
    """
    gamma = config.physical.gamma
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy

    Ny, Nx = Hc.shape
    N = Ny * Nx
    b = np.zeros(N)
    gamma = np.ones((Ny, Nx)) * gamma
    gamma_dt = gamma * dt

    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            Hc_val = Hc[j, i]
            term = Hc_val

            # Fluxes in x-direction
            if i < Nx - 1:
                denom = H_u[j, i + 1] + gamma_dt[j, i + 1] if H_u[j, i + 1] + gamma_dt[j, i + 1] != 0 else 1e-6
                coeff = (H_u[j, i + 1] * Gu[j, i + 1]) / denom
                term -= (dt / dx) * coeff
            if i > 0:
                denom = H_u[j, i] + gamma_dt[j, i] if H_u[j, i] + gamma_dt[j, i] != 0 else 1e-6
                coeff = (H_u[j, i] * Gu[j, i]) / denom
                term += (dt / dx) * coeff

            # Fluxes in y-direction
            if j < Ny - 1:
                denom = H_v[j + 1, i] + gamma_dt[j + 1, i] if H_v[j + 1, i] + gamma_dt[j + 1, i] != 0 else 1e-6
                coeff = (H_v[j + 1, i] * Gv[j + 1, i]) / denom
                term -= (dt / dy) * coeff
            if j > 0:
                denom = H_v[j, i] + gamma_dt[j, i] if H_v[j, i] + gamma_dt[j, i] != 0 else 1e-6
                coeff = (H_v[j, i] * Gv[j, i]) / denom
                term += (dt / dy) * coeff

            b[idx] = term

    return b

def update_velocity(config, Hc: np.ndarray, H_u: np.ndarray, H_v: np.ndarray, h: np.ndarray, 
                    u: np.ndarray, v: np.ndarray,
                    eta_new: np.ndarray, Fu: np.ndarray, Fv: np.ndarray) -> tuple:
    """
    Update the velocities u and v using the momentum equations.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        u: Velocity in x-direction at u-grid points (Ny, Nx + 1)
        v: Velocity in y-direction at v-grid points (Ny + 1, Nx)
        eta_new: Updated surface elevation at cell centers (Ny, Nx)
        Fu: Explicit term for u-velocity (Ny, Nx + 1)
        Fv: Explicit term for v-velocity (Ny + 1, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        gamma_T: Wind stress coefficient
        u_a: Wind speed in x-direction
        v_a: Wind speed in y-direction

    Returns:
        u_new: Updated velocity in x-direction at u-grid points (Ny, Nx + 1)
        v_new: Updated velocity in y-direction at v-grid points (Ny + 1, Nx)
    """
    g = config.physical.g
    gamma = config.physical.gamma
    gamma_T = config.physical.gamma_T
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    v_a = config.physical.v_a
    u_a = config.physical.u_a
    Ny, Nx = Hc.shape
    u_new = np.copy(u)
    v_new = np.copy(v)
    gamma = np.ones((Ny, Nx)) * gamma
    bag_grid = config.bag
    # bag_grid = np.zeros((Ny, Nx))  # Initialize the grid with zeros

    # # Find the middle indices
    # midx = Nx // 2
    # midy = Ny // 2
    # # Build a vertical wall in the middle (with some width)
    # bag_grid[:, midx-25:midx-15] = 1
    # bag_grid[midy-30:midy-5, :]  = 0  


    # Update u-velocity
    for j in range(Ny):
        for i in range(1, Nx):
            if H_u[j, i] > 0 and bag_grid[j, i-1] == 0:
            # if (H_u[j, i] - h[j, i]) > 0:
                eta_diff = eta_new[j, i] - eta_new[j, i - 1]
                friction = (gamma_T * u_a - gamma[j, i - 1] * u[j, i]) / H_u[j, i]
                u_new[j, i] = Fu[j, i] - g * dt / dx * eta_diff + dt * friction
            else:
                u_new[j, i] = 0.0

    # Update v-velocity
    for j in range(1, Ny):
        for i in range(Nx):
            if H_v[j, i]  > 0 and bag_grid[j-1, i] == 0:
            # if (H_v[j, i] - h[j,i]) > 0:
                eta_diff = eta_new[j, i] - eta_new[j - 1, i]
                friction = (gamma_T * v_a - gamma[j - 1, i] * v[j, i]) / H_v[j, i]
                v_new[j, i] = Fv[j, i] - g * dt / dy * eta_diff + dt * friction
            else:
                v_new[j, i] = 0.0

    return u_new, v_new

def update_velocity_vectorized( config, 
        Hc: np.ndarray, 
        H_u: np.ndarray, H_v: np.ndarray, 
        h: np.ndarray, 
        u: np.ndarray, v: np.ndarray,
        eta_new: np.ndarray, 
        Fu: np.ndarray, Fv: np.ndarray) -> tuple:
    """
    Update the velocities u and v using the momentum equations.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        u: Velocity in x-direction at u-grid points (Ny, Nx + 1)
        v: Velocity in y-direction at v-grid points (Ny + 1, Nx)
        eta_new: Updated surface elevation at cell centers (Ny, Nx)
        Fu: Explicit term for u-velocity (Ny, Nx + 1)
        Fv: Explicit term for v-velocity (Ny + 1, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        gamma_T: Wind stress coefficient
        u_a: Wind speed in x-direction
        v_a: Wind speed in y-direction

    Returns:
        u_new: Updated velocity in x-direction at u-grid points (Ny, Nx + 1)
        v_new: Updated velocity in y-direction at v-grid points (Ny + 1, Nx)
    """
    g = config.physical.g
    gamma = config.physical.gamma
    gamma_T = config.physical.gamma_T
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    v_a = config.physical.v_a
    u_a = config.physical.u_a
    Ny, Nx = Hc.shape

    # Initialize new velocity arrays
    u_new = np.copy(u)
    v_new = np.copy(v)
    gamma = np.ones((Ny, Nx)) * gamma
    bag_grid = config.bag

    # Update u-velocity using vectorized approach
    eta_diff_u = np.zeros_like(H_u)
    eta_diff_u[:, 1:-1] = eta_new[:, 1:] - eta_new[:, :-1]
    friction_u = (gamma_T * u_a - gamma[:, :] * u[:, 1:]) / H_u[:, 1:]
    friction_u = np.where(H_u[:, 1:] > 0, friction_u, 0.0)
    u_new[:, 1:] = np.where((H_u[:, 1:] > 0) & (bag_grid[:, :] == 0),
                Fu[:, 1:] - g * dt / dx * eta_diff_u[:, 1: ] + dt * friction_u, 0.0)

    # Update v-velocity using vectorized approach
    eta_diff_v = np.zeros_like(H_v)
    eta_diff_v[1:-1, :] = eta_new[1:, :] - eta_new[:-1, :]
    friction_v = (gamma_T * v_a - gamma[:, :] * v[1:, :]) / H_v[1:, :]
    friction_v = np.where(H_v[1:, :] > 0, friction_v, 0.0)
    v_new[1:, :] = np.where((H_v[1:, :] > 0) & (bag_grid[:, :] == 0),
                Fv[1:, :] - g * dt / dy * eta_diff_v[ 1:, :] + dt * friction_v, 0.0)

    return u_new, v_new

def update_velocity_v3(config, 
        Hc: np.ndarray, 
        H_u: np.ndarray, H_v: np.ndarray, 
        h: np.ndarray, 
        u: np.ndarray, v: np.ndarray,
        eta_new: np.ndarray, 
        Fu: np.ndarray, Fv: np.ndarray) -> tuple:
    """
    Update the velocities u and v using the momentum equations.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        u: Velocity in x-direction at u-grid points (Ny, Nx + 1)
        v: Velocity in y-direction at v-grid points (Ny + 1, Nx)
        eta_new: Updated surface elevation at cell centers (Ny, Nx)
        Fu: Explicit term for u-velocity (Ny, Nx + 1)
        Fv: Explicit term for v-velocity (Ny + 1, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        gamma_T: Wind stress coefficient
        u_a: Wind speed in x-direction
        v_a: Wind speed in y-direction

    Returns:
        u_new: Updated velocity in x-direction at u-grid points (Ny, Nx + 1)
        v_new: Updated velocity in y-direction at v-grid points (Ny + 1, Nx)
    """
    g = config.physical.g
    gamma = config.physical.gamma
    gamma_T = config.physical.gamma_T
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy
    v_a = config.physical.v_a
    u_a = config.physical.u_a
    Ny, Nx = Hc.shape

    # Initialize new velocity arrays
    u_new = np.copy(u)
    v_new = np.copy(v)
    gamma = np.ones((Ny, Nx)) * gamma
    bag_grid = config.bag

    # Update u-velocity using vectorized approach
    eta_diff_u = np.zeros_like(H_u)
    eta_diff_u[:, 1:-1] = eta_new[:, 1:] - eta_new[:, :-1]
    friction_u = (gamma_T * u_a - gamma[:, :] * u[:, 1:]) / H_u[:, 1:]
    friction_u = np.where(H_u[:, 1:] > 0, friction_u, 0.0)
    u_new[:, 1:] = np.where((H_u[:, 1:] > 0) & (bag_grid[:, :] == 0),
        Fu[:, 1:] - g * dt / dx * eta_diff_u[:, 1: ] + dt * friction_u, 0.0)

    # Update v-velocity using vectorized approach
    eta_diff_v = np.zeros_like(H_v)
    eta_diff_v[1:-1, :] = eta_new[1:, :] - eta_new[:-1, :]
    friction_v = (gamma_T * v_a - gamma[:, :] * v[1:, :]) / H_v[1:, :]
    friction_v = np.where(H_v[1:, :] > 0, friction_v, 0.0)
    v_new[1:, :] = np.where((H_v[1:, :] > 0) & (bag_grid[:, :] == 0),
        Fv[1:, :] - g * dt / dy * eta_diff_v[ 1:, :] + dt * friction_v, 0.0)

    return u_new, v_new

def update_H(config,Hc_old: np.ndarray, H_u: np.ndarray, H_v: np.ndarray, h: np.ndarray,
             u_new: np.ndarray, v_new: np.ndarray) -> np.ndarray:
    """
    Update the water depth H at cell centers based on new velocities.

    Parameters:
        Hc_old: Water depth at cell centers at time n (Ny, Nx)
        H_u: Water depth at u-grid points at time n (Ny, Nx + 1)
        H_v: Water depth at v-grid points at time n (Ny + 1, Nx)
        u_new: Updated velocity in x-direction at u-grid points (Ny, Nx + 1)
        v_new: Updated velocity in y-direction at v-grid points (Ny + 1, Nx)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        dt: Time step

    Returns:
        Hc_new: Updated water depth at cell centers at time n+1 (Ny, Nx)
    """
    Ny, Nx = Hc_old.shape
    dx = config.grid.dx
    dy = config.grid.dy
    dt = config.time.dt
    edge = 1e4 

    # Fluxes in x-direction
    flux_x = np.zeros((Ny, Nx + 1))
    flux_x[:, :] = H_u[:, :] * u_new[:, :]

    net_flux_x = (flux_x[:, 1:] - flux_x[:, :-1]) / dx

    # Fluxes in y-direction
    flux_y = np.zeros((Ny + 1, Nx))
    flux_y[:, :] = H_v[:, :] * v_new[:, :]

    net_flux_y = (flux_y[1:, :] - flux_y[:-1, :]) / dy

    # Update Hc_new
    Hc_new = Hc_old - dt * (net_flux_x + net_flux_y)
    # Hc_new = np.maximum(Hc_new, 0.0)

    Hc_new = np.clip(Hc_new, h, edge)
    return Hc_new

    
def matrix_T_vectorized(config,Hc: np.ndarray) -> csr_matrix:
    """
    Assemble the matrix T for solving eta_new.
    """
    Ny, Nx = Hc.shape
    N = Ny * Nx
    dx = config.grid.dx
    dy = config.grid.dy
    g = config.physical.g
    gamma = config.physical.gamma
    gamma = np.ones((Ny, Nx)) * gamma
    dt = config.time.dt

    gamma_dt = gamma * dt
    dt2_dx2 = (dt ** 2) / dx ** 2
    dt2_dy2 = (dt ** 2) / dy ** 2

    Hc_flat = Hc.ravel()
    gamma_dt_flat = gamma_dt.ravel()

    denom = Hc_flat + gamma_dt_flat
    denom[denom == 0] = 1e-6
    coeff = Hc_flat**2 / denom

    diagonal = Hc_flat + g * coeff * (dt2_dx2 + dt2_dy2)

    # Indices for the main diagonal
    row_indices = np.arange(N)
    col_indices = np.arange(N)
    data = diagonal.copy()

    # Helper arrays for indexing
    i = np.tile(np.arange(Nx), Ny)
    j = np.repeat(np.arange(Ny), Nx)
    idx = j * Nx + i

    # Off-diagonals
    # Left neighbor
    mask = i > 0
    row_indices = np.concatenate([row_indices, idx[mask]])
    col_indices = np.concatenate([col_indices, idx[mask] - 1])
    data = np.concatenate([data, -g * coeff[mask] * dt2_dx2])

    # Right neighbor
    mask = i < Nx - 1
    row_indices = np.concatenate([row_indices, idx[mask]])
    col_indices = np.concatenate([col_indices, idx[mask] + 1])
    data = np.concatenate([data, -g * coeff[mask] * dt2_dx2])

    # Bottom neighbor
    mask = j > 0
    row_indices = np.concatenate([row_indices, idx[mask]])
    col_indices = np.concatenate([col_indices, idx[mask] - Nx])
    data = np.concatenate([data, -g * coeff[mask] * dt2_dy2])

    # Top neighbor
    mask = j < Ny - 1
    row_indices = np.concatenate([row_indices, idx[mask]])
    col_indices = np.concatenate([col_indices, idx[mask] + Nx])
    data = np.concatenate([data, -g * coeff[mask] * dt2_dy2])

    # Assemble the sparse matrix
    T = coo_matrix((data, (row_indices, col_indices)), shape=(N, N)).tocsr()

    return T

def matrix_F_vectorized(config, u: np.ndarray, v: np.ndarray):

    """
    Compute the Finite Difference Operator F for both u and v using vectorized operations.
    """
    dx = config.grid.dx
    dy = config.grid.dy
    nu = config.physical.nu
    dt = config.time.dt
    # Initialize Fu and Fv
    Fu = np.zeros_like(u)
    Fv = np.zeros_like(v)
    
    # Define slices for the inner domain
    j_slice = slice(1, -1)
    i_slice = slice(1, -1)
    
    # Slices for u-component
    u_center = u[j_slice, i_slice]
    u_left   = u[j_slice, 0:-2]
    u_right  = u[j_slice, 2:]
    u_up     = u[0:-2, i_slice]
    u_down   = u[2:, i_slice]

    v_center = v[1:-2, :-1]
    
    # Compute advective term for u-component
    adv_u = (
        (u_center * (u_center - u_left) / dx ) + 
        (v_center * (u_center - u_up) / dy)
    )
    
    # Compute viscous term for u-component
    visc_u = nu * (
        (u_right - 2 * u_center + u_left) / dx**2 +
        (u_down - 2 * u_center + u_up) / dy**2
    )
    
    # Update Fu for u-component
    Fu[j_slice, i_slice] = u_center + dt * (-adv_u + visc_u)
    
    # Slices for v-component
    v_center = v[j_slice, i_slice]
    v_left   = v[j_slice, 0:-2]
    v_right  = v[j_slice, 2:]
    v_up     = v[0:-2, i_slice]
    v_down   = v[2:, i_slice]

    u_center = u[1:, 1:-2]
    
    # Compute advective term for v-component
    adv_v = (
        (u_center * (v_center - v_left) / dx ) + 
        ( v_center * (v_center - v_up) / dy)
    )
    
    # Compute viscous term for v-component
    visc_v = nu * (
        (v_right - 2 * v_center + v_left) / dx**2 +
        (v_down - 2 * v_center + v_up) / dy**2
    )
    
    # Update Fv for v-component
    Fv[j_slice, i_slice] = v_center + dt * (-adv_v + visc_v)
    
    return Fu, Fv

def vector_b_vectorized(config, Hc: np.ndarray, H_u: np.ndarray, H_v: np.ndarray, Gu: np.ndarray, Gv: np.ndarray) -> np.ndarray:

    """
    Compute the right-hand side vector b for solving eta_new.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        H_u: Total water depth at u-grid points (Ny, Nx + 1)
        H_v: Total water depth at v-grid points (Ny + 1, Nx)
        Gu: Flux term for u-velocity (Ny, Nx + 1)
        Gv: Flux term for v-velocity (Ny + 1, Nx)
        gamma: Friction coefficient at cell centers (Ny, Nx)
        dt: Time step
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        b: Right-hand side vector b (N,)
    """
    gamma = config.physical.gamma
    dt = config.time.dt
    dx = config.grid.dx
    dy = config.grid.dy

    Ny, Nx = Hc.shape
    N = Ny * Nx
    b = np.zeros(N)
    gamma_dt = np.ones((Ny, Nx)) * gamma * dt

    # Create copies of Hc to modify based on the conditions
    term = np.copy(Hc)

    # Fluxes in x-direction (right and left)
    denom_right = H_u[:, 1:] + gamma_dt[:, :] 
    denom_right = np.where(denom_right != 0, denom_right, 1e-6)  # Prevent division by zero
    coeff_right = (H_u[:, 1:] * Gu[:, 1:]) / denom_right 
    term[ :, : ] -= dt / dx * coeff_right

    denom_left = H_u[:, :-1] + gamma_dt[:, :] 
    denom_left = np.where(denom_left != 0, denom_left, 1e-6)
    coeff_left = (H_u[:, :-1] * Gu[:, :-1]) / denom_left
    term[:,  ] += (dt / dx) * coeff_left

    # Fluxes in y-direction (up and down)
    denom_down = H_v[1:, :] + gamma_dt[:, :] 
    denom_down = np.where(denom_down != 0, denom_down, 1e-6)
    coeff_down = (H_v[1:, :] * Gv[1:, :]) / denom_down
    term[: , :] -= (dt / dy) * coeff_down

    denom_up = H_v[:-1, :] + gamma_dt[:, :] 
    denom_up = np.where(denom_up != 0, denom_up, 1e-6)
    coeff_up = (H_v[:-1, :] * Gv[:-1, :]) / denom_up
    term[ :, :] += (dt / dy) * coeff_up

    # Flatten the 2D-term array to store in 1D-b array 
    b = term.flatten()

    return b 


def compute_fluxes(config, eta, u, v):
    g = config.physical.g
    # Calculate fluxes based on current state
        
    # print(f"eta.shape : {eta.shape}")
    # print(f"u.shape : {u.shape}")
    # print(f"v.shape : {v.shape}")

    F_hx = eta * u
    F_hy = eta * v
    F_ux = u**2 + 0.5 * g * eta**2
    F_vy = v**2 + 0.5 * g * eta**2

    # print(f"F_hx.shape : {F_hx.shape}")
    # print(f"F_hy.shape : {F_hy.shape}")
    # print(f"F_ux.shape : {F_ux.shape}")
    # print(f"F_vy.shape : {F_vy.shape}")

    return F_hx, F_hy, F_ux, F_vy

def saint_venant_update_vectorized_bv(config, Hc, eta, u, v, dt, dx, dy):

    h = Hc - eta
    F_hx, F_hy, F_ux, F_vy = compute_fluxes(config, eta, u, v)

    # print(f"F_hx.shape : {F_hx.shape}")
    # print(f"F_hy.shape : {F_hy.shape}")
    # print(f"F_ux[:, 1:].shape : {F_ux[:, 1:].shape}")
    # print(f"F_ux[:, :-1].shape : {F_ux[:, :-1].shape}")
    # print(f"F_vy[1:, :].shape : {F_vy[1:, :].shape}")
    # print(f"F_vy[:-1, :].shape : {F_vy[:-1, :].shape}")

    # Update velocities using finite difference method
    u[:, 1: ] -= dt / dx * (F_ux[:, 1:] - F_ux[:, :-1])
    v[1: , :] -= dt / dy * (F_vy[1:, :] - F_vy[:-1, :])

    # Update depth
    eta[:, 1: ] -= dt / dx * (F_hx[:, 1:] - F_hx[:, :-1])
    eta[1: , :] -= dt / dy * (F_hy[1:, :] - F_hy[:-1, :])

    Hc = h + eta 

    return Hc, eta, u, v

def saint_venant_update_vectorized(config, h, u, v, dt, dx, dy):

    F_hx, F_hy, F_ux, F_vy = compute_fluxes(config, h, u, v)

    # Update velocities using finite difference method
    u[:, 1:-1] -= dt / dx * (F_ux[:, 1:] - F_ux[:, :-1])
    v[1:-1, :] -= dt / dy * (F_vy[1:, :] - F_vy[:-1, :])

    # Update depth
    h[:, 1:-1] -= dt / dx * (F_hx[:, 1:] - F_hx[:, :-1])
    h[1:-1, :] -= dt / dy * (F_hy[1:, :] - F_hy[:-1, :])

    return h, u, v

def shallow_water_displacement(config, z, e, h) :
    """
        Compute new water levels 

    Arguments / Input 

        config: 
        z: nd-array     ground level          [m] 
        e: nd-array     eta / inundation      [m] 
        h: z + e        water surface level   [m] 

    Returns (updated/next iteration) : 

        z: nd-array     ground level          [m] 
        e: nd-array     eta / inundation      [m] 
        h: z + e        water surface level   [m] 
        u: nd-array     speed in x direction  [m/s] 
        v: nd-array     speed in y direction  [m/s] 

    """
    g  = config.physical.g
    sf = config.time.sf             #  sub-sample factor 
    ss_dt = config.time.dt / sf     #  sub-sample time in [s] 
    dx = config.grid.dx
    dy = config.grid.dy
    Sx = config.grid.Sx
    Sy = config.grid.Sy
    Sd = config.grid.Sd
    Bg = config.bag

    ee_clip = 100 

    #  print(f"sub-sample dt : {ss_dt}")
    factorX = 2 * ss_dt / dx 
    factorY = 2 * ss_dt / dy 
    flower = config.physical.flower

    Ny, Nx = z.shape
    Ny += 2             #  expand with a border at both sides
    Nx += 2             #  expand with a border at both sides
    CELLS = Ny * Nx

    CC_ZZ = np.zeros((Ny,Nx), dtype=np.float32)  # Ground Level ( h or z ) 
    CC_EE = np.zeros((Ny,Nx), dtype=np.float32)  # eta (Inundation Level) 
    CC_HH = np.zeros((Ny,Nx), dtype=np.float32)  # H ( water level = h + eta )
    CX_DH = np.zeros((Ny,Nx), dtype=np.float32)  # C -> X (east-west) flow dir 
    CY_DH = np.zeros((Ny,Nx), dtype=np.float32)  # C -> Y (south-north) flow dir 
    CX_FS = np.zeros((Ny,Nx), dtype=np.float32)  # C -> X (east-west)   flow speed 
    CY_FS = np.zeros((Ny,Nx), dtype=np.float32)  # C -> Y (south-north) flow speed  
    CX_HE = np.zeros((Ny,Nx), dtype=np.float32) 
    CX_HW = np.zeros((Ny,Nx), dtype=np.float32) 
    CY_HS = np.zeros((Ny,Nx), dtype=np.float32)
    CY_HN = np.zeros((Ny,Nx), dtype=np.float32)
    CC_GB = np.zeros((Ny,Nx), dtype=np.float32)   #  BAG info 
  
    #  Fill the inner part of mx with h
    CC_ZZ[ :, : ] = np.pad(z, pad_width=1, mode='edge')
    CC_EE[ :, : ] = np.pad(e, pad_width=1, mode='edge')
    
    #  inverse van BAG ( value = 0 when there is an obstakle )
    #  CC_GB[1:-1,1:-1] = 1 - np.copy(config.bag)
    #  CC_GB[ :, : ] = expand_edges( 1 - np.copy(config.bag) )
    CC_GB[  :, : ] = np.pad( 1 - config.bag, pad_width=1, mode='edge')
    CC_GB[  0, : ] = 0 #  Fill y =  0   row with 0 
    CC_GB[ -1, : ] = 0 #  Fill y = Ny-1 row with 0 
    CC_GB[  :, 0 ] = 0  
    CC_GB[  :, -1] = 0 

    #  Set eta (EE / Inundation) to 0 when BAG > 0 thus CC_CB == 0
    CC_EE[ :, : ] *= CC_GB[ :, : ]
    CC_EE[ :, : ] = np.clip(CC_EE[ : , : ], 0, ee_clip )

    CC_HH[:,:] = CC_ZZ[:,:] + CC_EE[:,:]
    
    #   define Flow Difference between 
    #   Current CELL <-> Eastern Adjacent Cell (both ways) 
    #   when positive => flow  from  Current CELL -> Eastern Adjacent
    #   when negative => flow  from  Eastern Adjacent -> Current CELL
    CX_DH[1:-1,1:-1] = CC_HH[1:-1,1:-1] - CC_HH[1:-1, 2:]

    # cc_hh_max = np.max( CC_HH )
    # cx_dh_max = np.max( CX_DH )

    # print(f"") 
    # print(f"cc_hh_max = {cc_hh_max}")
    # print(f"cx_dh_max = {cx_dh_max}")

    CX_HE[1:-1,1:-1] = np.minimum(
        np.abs(CX_DH[1:-1,1:-1]), CC_EE[1:-1, 1:-1] )
    

    CX_HW[1:-1,1:-1] = np.minimum(
        np.abs(CX_DH[1:-1,1:-1]), CC_EE[1:-1, 2:  ] )

    # cx_hw_max = np.max( CX_HW )
    # cx_he_max = np.max( CX_HE )

    # print(f"cx_hw_max = {cx_hw_max}")
    # print(f"cx_he_max = {cx_he_max}")


    CX_FS[1:-1,1:-1] = np.where(
        ( CX_DH[1:-1,1:-1] > 0 ),                # dir = east ? 
        ( +np.sqrt( g * flower * CX_HE[1:-1,1:-1] )), 
        ( -np.sqrt( g * flower * CX_HW[1:-1,1:-1] ))) 
    
    #  Set Speed = 0 if there is BAG ( building/obstacle)
    CX_FS[ :, : ] *= CC_GB[ : , : ]
       
    #   define Flow Difference  between 
    #   Cell Center <-> Southern Adjecent Cell (both ways)
    #   when positive  => flow from  Current CELL -> Southern Adjacent
    #   when negative  => flow from  Southern Adjacent => Current CELL
    CY_DH[1:-1,1:-1] = CC_HH[1:-1,1:-1] - CC_HH[2:,1:-1]

    CY_HS[1:-1,1:-1] = np.minimum(
        np.abs(CY_DH[1:-1,1:-1]), CC_EE[1:-1, 1:-1] )
    
    CY_HN[1:-1,1:-1] = np.minimum(
        np.abs(CY_DH[1:-1,1:-1]), CC_EE[2: , 1:-1] )

    CY_FS[1:-1,1:-1] = np.where (
        ( CY_DH[1:-1,1:-1] > 0 ),                # dir = south ?  
        ( +np.sqrt( g * flower * CY_HS[1:-1,1:-1] )), 
        ( -np.sqrt( g * flower * CY_HN[1:-1,1:-1] ))) 
    
    #  Set Speed = 0 if there is BAG ( building/obstacle)
    CY_FS[ :, : ] *= CC_GB[ : , : ]

    u = np.copy( CX_FS[1:-1,1:-1] ) 
    v = np.copy( CY_FS[1:-1,1:-1] )

    # print(f"CX_FS[sample_area]: \n{CX_FS[Sy-Sd+1:Sy+Sd+1,Sx-Sd+1:Sx+Sd+1]}\n")
    # print(f"CX_HE[sample_area]: \n{CX_HE[Sy-Sd+1:Sy+Sd+1,Sx-Sd+1:Sx+Sd+1]}\n")

    for s in range ( 1, sf + 1 , 1 ): 

        CC_HH[:,:] = CC_ZZ[:,:] + CC_EE[:,:]
        # #   define Flow Difference between 
        # #   Current CELL <-> Eastern Adjacent Cell (both ways) 
        # #   when positive => flow  from  Current CELL -> Eastern Adjacent
        # #   when negative => flow  from  Eastern Adjacent -> Current CELL
        CX_DH[1:-1,1:-1] = CC_HH[1:-1,1:-1] - CC_HH[1:-1, 2:]
        CX_HE[1:-1,1:-1] = np.minimum(
            np.abs(CX_DH[1:-1,1:-1]), CC_EE[1:-1, 1:-1] )
        CX_HW[1:-1,1:-1] = np.minimum(
            np.abs(CX_DH[1:-1,1:-1]), CC_EE[1:-1, 2:  ] )

        if config.mid_fs_update : 

            CX_FS[1:-1,1:-1] = np.where(
                ( CX_DH[1:-1,1:-1] > 0 ),                # dir = east ? 
                ( +np.sqrt( g * flower * CX_HE[1:-1,1:-1] )), 
                ( -np.sqrt( g * flower * CX_HW[1:-1,1:-1] ))) 
            
            CX_FS[ :, : ] *= CC_GB[ : , : ]

        # #   define Flow Difference  between 
        # #   Cell Center <-> Southern Adjecent Cell (both ways)
        # #   when positive  => flow from  Current CELL -> Southern Adjacent
        # #   when negative  => flow from  Southern Adjacent => Current CELL
        CY_DH[1:-1,1:-1] = CC_HH[1:-1,1:-1] - CC_HH[2:,1:-1]
        CY_HS[1:-1,1:-1] = np.minimum(
            np.abs(CY_DH[1:-1,1:-1]), CC_EE[1:-1, 1:-1] )
        CY_HN[1:-1,1:-1] = np.minimum(
            np.abs(CY_DH[1:-1,1:-1]), CC_EE[2: , 1:-1] )

        if config.mid_fs_update : 
            
            CY_FS[1:-1,1:-1] = np.where (
                ( CY_DH[1:-1,1:-1] > 0 ),                # dir = south ?  
                ( +np.sqrt( g * flower * CY_HS[1:-1,1:-1] )), 
                ( -np.sqrt( g * flower * CY_HN[1:-1,1:-1] ))) 
            
            CY_FS[ :, : ] *= CC_GB[ : , : ]
    
        #  Next portion of code is to randomze flow direction 
        #  
        #  Initialize an array to mark used numbers (False means unused)
        used_numbers = [False,False,False,False]
 
        # Number of unique numbers to generate
        total_directions = 4
        numbers_generated = 0

        while numbers_generated < total_directions:
            # Generate a random number between 1 and 4
            random_number = random.randint(1,4)
            
            #   Check if the number has already been used
            #   If the number has been used,continue the loop 
            #   without incrementing numbers_generated
            if not used_numbers[random_number - 1]:
                # Mark the number as used
                used_numbers[random_number - 1] = True
                numbers_generated += 1
                # Output the number
                # print(f"Random Number:{random_number}")

                if random_number == 1: 
                    #  to east --> 
                    CC_EE[1:-1,2:  ] += np.where(
                        ( CC_EE[1:-1,1:-1] > 0 ) & ( CX_FS[1:-1,1:-1] > 0) , 
                        ( CX_FS[1:-1,1:-1] * CX_HE[1:-1,1:-1] * factorX ),
                        0 )
                    
                    CC_EE[1:-1,1:-1] -= np.where(
                        ( CC_EE[1:-1,1:-1] > 0 ) & ( CX_FS[1:-1,1:-1] > 0), 
                        ( CX_FS[1:-1,1:-1] * CX_HE[1:-1,1:-1] * factorX ), 
                        0 )

                elif random_number == 2:
                    #  from east 
                    CC_EE[1:-1,1:-1] -= np.where(
                        ( CC_EE[1:-1,2:  ] > 0 ) & ( CX_FS[1:-1,1:-1] < 0), 
                        ( CX_FS[1:-1,1:-1] * CX_HW[1:-1,1:-1] * factorX ), 
                        0 )
                    
                    CC_EE[1:-1,2:  ] += np.where(
                        ( CC_EE[1:-1,2:  ] > 0 ) & ( CX_FS[1:-1,1:-1] < 0) , 
                        ( CX_FS[1:-1,1:-1] * CX_HW[1:-1,1:-1] * factorX ),
                        0 )
                    
                elif random_number == 3:
                    #  to south  --> 
                    CC_EE[2:  ,1:-1] += np.where(
                        ( CC_EE[1:-1,1:-1] > 0 ) & ( CY_FS[1:-1,1:-1] > 0) , 
                        ( CY_FS[1:-1,1:-1] * CY_HS[1:-1,1:-1] * factorY ),
                        0 )
                    
                    CC_EE[1:-1,1:-1] -= np.where(
                        ( CC_EE[1:-1,1:-1] > 0 ) & ( CY_FS[1:-1,1:-1] > 0), 
                        ( CY_FS[1:-1,1:-1] * CY_HS[1:-1,1:-1] * factorY ),
                        0 )

                elif random_number == 4:
                    #  from south  
                    CC_EE[1:-1,1:-1] -= np.where(
                        ( CC_EE[2:  ,1:-1] > 0 ) & ( CY_FS[1:-1,1:-1] < 0), 
                        ( CY_FS[1:-1,1:-1] * CY_HN[1:-1,1:-1] * factorY ),
                        0 )
                    
                    CC_EE[2:  , 1:-1] += np.where(
                        ( CC_EE[2:  , 1:-1]> 0 ) & ( CY_FS[1:-1,1:-1] < 0) , 
                        ( CY_FS[1:-1,1:-1] * CY_HN[1:-1,1:-1] * factorY ),
                        0 )
                
                CC_EE[ :, : ] *= CC_GB[ :, : ]
                CC_EE[ :, : ] = np.clip(CC_EE[ : , : ], 0, ee_clip ) 
                
        # print(f"Sub Sample {s} CC_EE[ sample_area]: \n{CC_EE[ 
        #     Sy-Sd+1:Sy+Sd+1,Sx-Sd+1:Sx+Sd+1]}\n")
        
        # CC_EE[:,:] = np.clip(CC_EE[:,:], 0, edge ) 

        #print(f"CC_EE: CC_EE[1:-1,1:-1]\n{CC_EE[1:-1,1:-1]}\n") 

        # e = CC_EE[1:-1,1:-1] 
        # print(f"sample: {s},  e[ full_area ]: \n{e}\n")
        # print(f"sample: {s},  e[ sample_area ]:\n{e[Sy-Sd+1:Sy+Sd+1,Sx-Sd+1:Sx+Sd+1]}\n")

        #  e = CC_EE[1:-1,1:-1] 
        # sum_e = np.sum( CC_EE[1:-1,1:-1] ) 
        # # max_z = np.max( CC_ZZ[1:-1,1:-1] )
        # max_e = np.max( CC_EE[1:-1,1:-1] )
        # # max_h = np.max( CC_HH[1:-1,1:-1] )

        # prc = 100 * s/sf
        # print(f"="*40) 

        # print(f"sub sample:{s} ({prc}%), max_e:{max_e:.6f}, sum_e:{sum_e:.3f}")
        # e = CC_EE[1:-1,1:-1]
        # print(f"    e[sample_area]:\n{e[Sy-Sd+1:Sy+Sd+1,Sx-Sd+1:Sx+Sd+1]}\n")
        
        # print(f"    Plot 3D area:\n") 
        # plot_3D_area(config, f"Inundation, sample:{s}", e)

    CC_EE[ :, : ] = np.clip(CC_EE[ : , : ], 0, ee_clip ) 
    CC_HH[:,:] = CC_ZZ[:,:] + CC_EE[:,:]
    z = np.copy( CC_ZZ[1:-1,1:-1] )
    e = np.copy( CC_EE[1:-1,1:-1] )
    h = np.copy( CC_HH[1:-1,1:-1] )
    u = np.copy( CX_FS[1:-1,1:-1] ) 
    v = np.copy( CY_FS[1:-1,1:-1] )
    return z, e, h, u, v 


def get_total_flow_1D(config, e: np.ndarray, m: np.ndarray, 
                   u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes the total flow (flux) and net flow for a staggered grid.
    
    Args:
        - e (2D array): Inundation (eta) at the center of each cell (ny, nx).
        - m (2D array): Measurement Cell ( 0=no, 1 = yes )
        - u (2D array): Velocity in x-direction
        - v (2D array): Velocity in y-direction
    
    Returns:
        float:  Total flow (in poitive direction) 
                in both X and Y direction (vector addition)

    """
    dx = config.grid.dx
    dy = config.grid.dy

    Qx_pos = np.zeros( e.shape)
    #  Qx_neg = np.zeros( e.shape)
    Qy_pos = np.zeros( e.shape)
    #  Qy_neg = np.zeros( e.shape)

    if config.flow_measure.dir == "x" or config.flow_measure.dir == "u": 
        Qx_pos = np.where( u > 0, e * m * u * dy, 0)  # Flow through cells - East
        #  Qx_neg = np.where( u < 0, e * m * u * dy, 0)  # Flow through cells - West 

    elif config.flow_measure.dir == "y" or config.flow_measure.dir == "v": 
        Qy_pos = np.where( v > 0, e * m * v * dx, 0)  # Flow through cells - South 
        #  Qy_neg = np.where( v < 0, e * m * v * dx, 0)  # Flow through cells - South 

    elif (  config.flow_measure.dir == "xy" or config.flow_measure.dir == "uv" or 
            config.flow_measure.dir == "yx" or config.flow_measure.dir == "vu" ) : 

        # Compute flow rates in x-direction (Qx) and y-direction (Qy)
        Qx_pos = np.where( u > 0, e * m * u * dy, 0)  # Flow through cells - East
        #  Qx_neg = np.where( u < 0, e * m * u * dy, 0)  # Flow through cells - West 
        Qy_pos = np.where( v > 0, e * m * v * dx, 0)  # Flow through cells - South 
        #  Qy_neg = np.where( v < 0, e * m * v * dx, 0)  # Flow through cells - South 

    # V = np.sqrt( np.square(Qx) + np.square(Qy))

    db_zM_pos = np.sum( np.sqrt( np.square(Qy_pos) + np.square(Qx_pos)) )

    return db_zM_pos

def get_total_flow_2D(config, e: np.ndarray, m: np.ndarray, 
                   u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes the total flow (flux) and net flow for a staggered grid.
    
    Args:
        - e (2D array): Inundation (eta) at the center of each cell (ny, nx).
        - m (2D array): Measurement Cell ( 0=no, 1 = yes )
        - u (2D array): Velocity in x-direction
        - v (2D array): Velocity in y-direction
    
    Returns:
        np.ndarray:  effective flow 

    """
    dx = config.grid.dx
    dy = config.grid.dy

    Qx = np.zeros( e.shape )
    #  Qx_neg = np.zeros( e.shape )
    Qy = np.zeros( e.shape )
    #  Qy_neg = np.zeros( e.shape )

    if config.flow_measure.dir == "x" or config.flow_measure.dir == "u": 
        Qx = e * m * u * dy  # Flow through cells - East
        #  Qx_neg = np.where( u < 0, e * u * dy, 0)  # Flow through cells - West 

    elif config.flow_measure.dir == "y" or config.flow_measure.dir == "v": 
        Qy = e * m * v * dx  # Flow through cells - South 
        #  Qy_neg = np.where( v < 0, e * v * dx, 0)  # Flow through cells - South 

    elif (  config.flow_measure.dir == "xy" or config.flow_measure.dir == "uv" or 
            config.flow_measure.dir == "yx" or config.flow_measure.dir == "vu" ) : 

        # Compute flow rates in x-direction (Qx) and y-direction (Qy)
        Qx = e * m * u * dy  # Flow through cells - East
        #   Qx_neg = np.where( u < 0, e * u * dy, 0)  # Flow through cells - West 
        Qy = e * m * v * dx  # Flow through cells - South 
        #   Qy_neg = np.where( v < 0, e * v * dx, 0)  # Flow through cells - South 

    # V = np.sqrt( np.square(Qx) + np.square(Qy))

    db_zM = np.sqrt( np.square(Qy) + np.square(Qx))
    #  db_zM_neg = np.sqrt( np.square(Qy_neg) + np.square(Qx_neg))
    
    return db_zM

