#srs\tools\check.py
import time
import numpy as np

def exec_time(method, *args, **kwargs):
    start_time = time.time()
    result = method(*args, **kwargs)
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    return elapsed_time_ms, result



def boundary_velocity(u: np.ndarray, v: np.ndarray):
    """
    Implement boundary conditions to ensure that the velocity fields are consistent with physical constraints.
    """
    # Set velocities at domain boundaries so outflow is consistent
    u[:, 0]  = u[:, 1]   # Left boundary
    u[:, -1] = u[:,-2]   # Right boundary
    u[:, -2] = u[:,-3]  #  u[:,-2]   # Right boundary
    v[0, :]  = v[1, :]   # Top boundary 
    v[-1, :] = v[-2, :]  # Bottom boundary
    v[-2, :] = v[-3, :] #  v[-2, :]  # Bottom boundary

    return u, v

def water_lvl_stable(eta_new: np.ndarray, threshold: float = 1e-4) -> bool:
    """
    Check if the water level has stabilized.

    Parameters:
        eta_new: Surface elevation at cell centers (Ny, Nx)
        threshold: Relative standard deviation threshold

    Returns:
        True if water level is stable, False otherwise
    """
    mean_eta = np.mean(eta_new)
    std_eta = np.std(eta_new)

    if mean_eta == 0:
        return std_eta < threshold

    relative_std = std_eta / abs(mean_eta)
    return relative_std < threshold

def flow_stable(u: np.ndarray, v: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if the flow velocities are negligible across the grid.

    Parameters:
    - u: Velocity in the x-direction
    - v: Velocity in the y-direction
    - threshold: Maximum absolute velocity to consider flow negligible

    Returns:
    - True if all velocities are below the threshold, False otherwise
    """
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))

    return max_u < threshold and max_v < threshold

def water_lvl_treshold(eta_old: np.ndarray, eta_new: np.ndarray, threshold: float) -> bool:
    """
    Check if less than 1% of the grid points in eta have changed significantly.

    Parameters:
        eta_old: Previous surface elevation at cell centers (Ny, Nx)
        eta_new: Current surface elevation at cell centers (Ny, Nx)
        threshold: Relative change threshold for considering a point as changed

    Returns:
        True if less than 1% of the grid points have changed, False otherwise.
    """
     # Calculate absolute difference
    diff = np.abs(eta_new - eta_old)
    
    # Compute max values to handle relative changes
    max_eta = np.maximum(np.abs(eta_old), np.abs(eta_new))
    
    # Avoid division by zero by using a small epsilon where max_eta is very close to 0
    epsilon = 1e-10
    max_eta_safe = np.where(max_eta > epsilon, max_eta, epsilon)
    
    # Calculate relative change safely
    relative_change = diff / max_eta_safe
    
    # Count points where the relative change is greater than the threshold
    changed_points = np.sum(relative_change > threshold)
    total_points = eta_new.size

    # Check if more than 99% of the points have changed less than the threshold
    return changed_points / total_points < 0.01

def flow_treshold(u_old: np.ndarray, u_new: np.ndarray, 
            v_old: np.ndarray, v_new: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if less than 1% of the grid points in flow velocities have changed significantly.

    Parameters:
    - u_old: Previous velocity in the x-direction
    - v_old: Previous velocity in the y-direction
    - u_new: Current velocity in the x-direction
    - v_new: Current velocity in the y-direction
    - threshold: Relative change threshold for considering a point as changed

    Returns:
    - True if less than 1% of the grid points have changed, False otherwise
    """
    # Calculate absolute difference for u and v
    diff_u = np.abs(u_new - u_old)
    diff_v = np.abs(v_new - v_old)

    # Compute max values to avoid dividing by zero for relative changes
    max_u = np.maximum(np.abs(u_old), np.abs(u_new))
    max_v = np.maximum(np.abs(v_old), np.abs(v_new))

    # Avoid division by zero and calculate relative change
    epsilon = 1e-10
    max_u_safe = np.where(max_u > epsilon ,  max_u, epsilon)
    max_v_safe = np.where(max_v > epsilon ,  max_v, epsilon)

    relative_change_u = diff_u / max_u_safe
    relative_change_v = diff_v / max_v_safe

    # Count how many points have changed more than the threshold
    changed_points_u = np.sum(relative_change_u > threshold)
    changed_points_v = np.sum(relative_change_v > threshold)
    total_points = u_new.size  # u and v should have the same size

    # Check if more than 99% of the points are below the threshold for both u and v
    return (changed_points_u / total_points < 0.01) and (changed_points_v / total_points < 0.01)

def mass_conservation(H: np.ndarray, u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> bool:
    return

def continuity_check(H: np.ndarray, u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> bool:
    return

def CFL_condition_check(H: np.ndarray, u: np.ndarray, v: np.ndarray, g: float, dt: float, dx: float, dy: float) -> bool:
    """ Check CFL condition for stability """
    celerity = np.sqrt(g * H)
    max_velocity = np.maximum(np.abs(u[:, :-1]), np.abs(v[:-1, :]))
    cfl_x = (dt / dx) * (max_velocity / celerity)
    cfl_y = (dt / dy) * (max_velocity / celerity)
    return np.all(cfl_x < 1) and np.all(cfl_y < 1)