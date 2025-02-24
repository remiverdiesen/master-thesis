# src/solver.py

import numpy as np
import logging
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg, gmres, spsolve
from .config import Config

def newton_solver(config:Config, Hc: np.ndarray, eta: np.ndarray, h: np.ndarray, b: np.ndarray,
                  T: csr_matrix) -> np.ndarray:
    """
    Newton's method with damping for solving the surface elevation.

    Args:
        Hc:     Total water depth at cell centers 
        eta:    Current surface elevation 
        h:      Bottom topography at cell centers 
        b:      Right-hand side vector (N,)
        T:      Sparse matrix T (N, N)
        tol:    Convergence tolerance
        max_iter: Maximum number of iterations

    Returns:
        eta_new: Updated surface elevation after convergence (Ny, Nx)
    """
    logging.debug("Starting Newton Iteration to solve for surface elevation.")

    Ny, Nx = Hc.shape
    Nxy = Ny * Nx
    h_flat = h.flatten()
    eta_flat = eta.flatten()

    # Initial guess zeta^0 > -h
    zeta = eta_flat.copy() + config.solver.epsilon

    for iter in range(config.solver.max_iteration):
        logging.debug(f"    Newton Iteration {iter}:")
        # Construct the diagonal matrix D^l
        D_data = np.where(h_flat + zeta > 0, 1.0, 0.0)
        D = diags(D_data, format='csr')

        # Compute H(ζ^l)
        H_zeta = np.maximum(h_flat, h_flat + zeta)

        # Compute residual r^l = H(ζ^l) + T ζ^l - b
        residual = H_zeta + T.dot(zeta) - b

        # Assemble A = D^l + T
        A = D + T

        # Solve A δ^l = residual 
        delta, info = gmres(A, residual, rtol=config.solver.tolerance) # cg(A, residual, rtol=tole)

        # Apply damping
        alpha = 0.5  # Damping factor between 0 and 1
        zeta_new = zeta - alpha * delta

        # Check for convergence
        delta_norm = np.linalg.norm(delta)
        residual_norm = np.linalg.norm(residual)
        logging.debug(f"        Residual norm = {residual_norm}")
        logging.debug(f"        Delta norm    = {delta_norm}")

        if delta_norm < config.solver.tolerance:
            logging.info(f"Solver converged in {iter} iterations.")
            eta_new = zeta_new.reshape((Ny, Nx))
            # eta_new = np.clip(eta_new, 0, np.inf )
            return eta_new

        zeta = zeta_new.copy()

    # If max iterations are reached without convergence
    eta_new = zeta.reshape((Ny, Nx))
    # eta_new = np.clip(eta_new, 0, np.inf )
    logging.info("Reached maximum iterations without full convergence.")
    return eta_new


def newton_solver_modified_trap(config:Config, Hc: np.ndarray, eta: np.ndarray, h: np.ndarray, b: np.ndarray,
                  T: csr_matrix) -> np.ndarray:
    """
    Newton's method with damping for solving the surface elevation.

    Args:
        Hc:     Total water depth at cell centers 
        eta:    Current surface elevation 
        h:      Bottom topography at cell centers 
        b:      Right-hand side vector (N,)
        T:      Sparse matrix T (N, N)
        tol:    Convergence tolerance
        max_iter: Maximum number of iterations

    Returns:
        eta_new: Updated surface elevation after convergence (Ny, Nx)
    """
    logging.debug("Starting Newton Iteration to solve for surface elevation.")

    Ny, Nx = Hc.shape
    N = Ny * Nx
    h_flat = h.flatten()
    eta_flat = eta.flatten()

    # Initial guess zeta^0 > -h
    zeta = eta_flat.copy() + config.solver.epsilon

    for iter in range(config.solver.max_iteration):
        logging.debug(f"    Iteration {iter + 1}:")
        # Construct the diagonal matrix D^l
        D_data = np.where(h_flat + zeta > 0, 1.0, 0.0)
        D = diags(D_data, format='csr')

        # Compute H(ζ^l)
        H_zeta = np.maximum(h_flat, h_flat + zeta)

        # Compute residual r^l = H(ζ^l) + T ζ^l - b
        residual = H_zeta + T.dot(zeta) - b

        # Assemble A = D^l + T
        A = D + T

        # Solve A δ^l = residual 
        delta, info = gmres(A, residual, rtol=config.solver.tolerance) 
        # delta, info = cg(A, residual, rtol=config.solver.tolerance)

        # Apply damping
        alpha = 0.5  # Damping factor between 0 and 1 
        zeta_pred = zeta - alpha * delta
        zeta_new = 0.5 * (zeta + zeta_pred)  # Averaging step for stability (Modified Trapezoidal)

        # Check for convergence
        delta_norm = np.linalg.norm(delta)
        residual_norm = np.linalg.norm(residual)
        logging.debug(f"        Residual norm = {residual_norm}")
        logging.debug(f"        Delta norm    = {delta_norm}")

        if delta_norm < config.solver.tolerance:
            eta_new = zeta_new.reshape((Ny, Nx))
            logging.info(f"Solver converged in {iter + 1} iterations.")
            return eta_new

        zeta = zeta_new.copy()

    # If max iterations are reached without convergence
    eta_new = zeta.reshape((Ny, Nx))
    logging.info("Reached maximum iterations without full convergence.")
    return eta_new


"""
--------------------------------------------------------------------------

Below methods are for the parallel implementation of the Newton solver
and not used...yet!
---------------------------------------------------------------------

"""

# @njit(parallel=True)
def compute_H_zeta(h_flat: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    Compute H(zeta) = max(0, h + zeta).

    Parameters:
        h_flat: Flattened bottom topography array.
        zeta: Current zeta array.

    Returns:
        H_zeta: Computed H(zeta) array.
    """
    H_zeta = np.maximum(0.0, h_flat + zeta)
    return H_zeta

# @njit(parallel=True)
def compute_D_data(h_flat: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal entries of matrix D.

    Parameters:
        h_flat: Flattened bottom topography array.
        zeta: Current zeta array.

    Returns:
        D_data: Diagonal data for matrix D.
    """
    N = zeta.size
    D_data = np.zeros(N)
    for i in range(N):
        D_data[i] = 1.0 if h_flat[i] + zeta[i] > 0 else 0.0
    return D_data

def newton_solver_parallel(Hc: np.ndarray, eta: np.ndarray, h: np.ndarray, b: np.ndarray,
                  T: csr_matrix, tolerance: float, max_iterations: int) -> np.ndarray:
    """
    Newton's method with PETSc for solving the surface elevation.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        eta: Current surface elevation (Ny, Nx)
        h: Bottom topography at cell centers (Ny, Nx)
        b: Right-hand side vector (N,)
        T: Sparse matrix T (N, N) in CSR format
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        eta_new: Updated surface elevation after convergence (Ny, Nx)
    """
    logging.debug("Starting Newton Iteration with PETSc to solve for surface elevation.")

    Ny, Nx = Hc.shape
    N = Ny * Nx
    h_flat = h.ravel()
    eta_flat = eta.ravel()

    # Initial guess zeta^0 > -h
    zeta = eta_flat.copy() + 1e-6

    # Convert T to PETSc matrix
    T_petsc = PETSc.Mat().createAIJ(size=T.shape, csr=(T.indptr, T.indices, T.data))
    T_petsc.assemblyBegin()
    T_petsc.assemblyEnd()

    # Prepare PETSc vectors
    b_petsc = PETSc.Vec().createWithArray(b)
    zeta_petsc = PETSc.Vec().createWithArray(zeta)

    for iteration in range(max_iterations):
        logging.debug(f"    Iteration {iteration + 1}:")
        # Compute D_data using Numba-accelerated function
        D_data = compute_D_data(h_flat, zeta)
        D = diags(D_data, format='csr')

        # Convert D to PETSc matrix
        D_petsc = PETSc.Mat().createAIJ(size=(N, N), csr=(D.indptr, D.indices, D.data))
        D_petsc.assemblyBegin()
        D_petsc.assemblyEnd()

        # Compute H(ζ^l) using Numba-accelerated function
        H_zeta = compute_H_zeta(h_flat, zeta)

        # Compute residual r^l = H(ζ^l) + T ζ^l - b
        Tzeta_petsc = T_petsc * zeta_petsc
        residual_array = H_zeta + Tzeta_petsc.getArray() - b
        residual_petsc = PETSc.Vec().createWithArray(residual_array)

        # Assemble A = D^l + T in PETSc
        A_petsc = D_petsc.copy()
        A_petsc.axpy(1.0, T_petsc)

        # Set up KSP solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType('cg')  # Change to 'gmres' if necessary
        pc = ksp.getPC()
        pc.setType('jacobi')  # Try different preconditioners as needed
        ksp.setTolerances(rtol=tolerance, max_it=1000)
        ksp.setFromOptions()

        delta_petsc = PETSc.Vec().createWithArray(np.zeros(N))

        ksp.solve(residual_petsc, delta_petsc)

        if not ksp.getConvergedReason():
            logging.warning(f"KSP did not converge at iteration {iteration + 1}.")
            break

        # Apply damping
        alpha = 0.5  # Damping factor between 0 and 1
        delta = delta_petsc.getArray()
        zeta_new = zeta - alpha * delta

        # Check for convergence
        delta_norm = np.linalg.norm(delta)
        residual_norm = np.linalg.norm(residual_array)
        logging.debug(f"        Residual norm = {residual_norm}")
        logging.debug(f"        Delta norm    = {delta_norm}")

        if delta_norm < tolerance:
            eta_new = zeta_new.reshape((Ny, Nx))
            logging.info(f"Solver converged in {iteration + 1} iterations.")
            return eta_new

        zeta = zeta_new.copy()
        zeta_petsc = PETSc.Vec().createWithArray(zeta)

    # If max iterations are reached without convergence
    eta_new = zeta.reshape((Ny, Nx))
    logging.warning("Reached maximum iterations without full convergence.")
    return eta_new

def picard_solver(Hc: np.ndarray, eta: np.ndarray, h: np.ndarray, b: np.ndarray,
                  T: csr_matrix, tolerance: float, max_iterations: int) -> np.ndarray:
    """
    Picard iteration for solving the surface elevation.

    Parameters:
        Hc: Total water depth at cell centers (Ny, Nx)
        eta: Current surface elevation (Ny, Nx)
        h: Bottom topography at cell centers (Ny, Nx)
        b: Right-hand side vector (N,)
        T: Sparse matrix T (N, N)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        eta_new: Updated surface elevation after convergence (Ny, Nx)
    """
    logging.debug("Starting Picard Iteration to solve for surface elevation.")

    Ny, Nx = Hc.shape
    N = Ny * Nx
    h_flat = h.flatten()
    eta_flat = eta.flatten()

    # Initial guess zeta^0
    zeta = eta_flat.copy()

    for iteration in range(max_iterations):
        logging.debug(f"    Iteration {iteration + 1}:")
        # Compute H(ζ^l)
        H_zeta = np.maximum(0.0, h_flat + zeta)

        # Compute the right-hand side
        rhs = b - H_zeta

        # Solve T ζ^{l+1} = rhs
        try:
            zeta_new = spsolve(T, rhs)
        except Exception as e:
            logging.error(f"Linear solver failed: {e}")
            raise

        # Check for convergence
        delta = zeta_new - zeta
        delta_norm = np.linalg.norm(delta)
        logging.debug(f"        Delta norm = {delta_norm}")

        if delta_norm < tolerance:
            eta_new = zeta_new.reshape((Ny, Nx))
            logging.info(f"Solver converged in {iteration + 1} iterations.")
            return eta_new

        zeta = zeta_new.copy()

    # If max iterations are reached without convergence
    eta_new = zeta.reshape((Ny, Nx))
    logging.warning("Reached maximum iterations without full convergence.")
    return eta_new


