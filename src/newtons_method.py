import numpy as np
import logging

logger = logging.getLogger(__name__)


def newtons_method(
    theta0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    f,
    grad,
    hessian,
    gap,
    eps,
    bt_factor,
    max_iter=50,
    verbose=False,
):
    """
    Performs newton's method with backtracking line search.

    Parameters
    ----------
    theta0
    X
    y
    f
    grad
    hessian
    gap
    eps
    bt_factor
    max_iter
    verbose

    Returns
    -------

    """
    # start = time.perf_counter()
    gap_k = gap(theta0, X, y)
    gaps = [gap_k]
    fs = []
    theta = theta0
    counter = 0
    is_converged = False

    # time_per_iteration = []
    while counter < max_iter:
        if gap_k < eps:
            is_converged = True
            break
        # start_per_iter = time.perf_counter()
        counter += 1
        if verbose:
            logger.info(f"Newton step: {counter}")
        f_k = f(theta, X, y)
        grad_k = grad(theta, X, y)
        hessian_k = hessian(theta, X, y)
        nu_k = -np.linalg.solve(hessian_k, grad_k)
        t = 1
        bt_counter = 0
        lhs = f(theta + t * nu_k, X, y)
        rhs = f_k + 0.5 * t * np.dot(grad_k, nu_k)
        while lhs > rhs:
            bt_counter += 1
            t = bt_factor * t
            lhs = f(theta + t * nu_k, X, y)
            rhs = f_k + 0.5 * t * np.dot(grad_k, nu_k)
        if verbose:
            logging.info(f"Number of backtracking steps: {bt_counter}")
        theta = theta + t * nu_k
        gap_k = gap(theta, X, y)
        gaps.append(gap_k)
        fs.append(f_k)
        if verbose:
            logging.info(f"Current f: {f_k}")
            logging.info(f"Current gap: {gap_k}")

        # end_per_iter = time.perf_counter()
        # time_per_iteration.append(round(end_per_iter-start_per_iter,3))
    # end = time.perf_counter()
    # if is_converged:
    # print(f"""Total iterations: {counter}
    # Total time: {end - start:.3f} seconds
    # Average time per iteration: {np.mean(time_per_iteration):.3f} seconds
    #        """)
    # else:
    # print("Convergence failed")
    # print(f"Newton steps: {counter}")
    return fs, gaps, is_converged, theta
