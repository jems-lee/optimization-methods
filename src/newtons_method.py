import numpy as np
import logging

logger = logging.getLogger(__name__)


def newtons_method(
    beta0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    f,
    grad,
    hessian,
    gap,
    eps,
    backtracking,
    bt_factor,
    verbose=False,
):
    # start = time.perf_counter()
    gap_k = gap(beta0, X, y)
    gaps = [gap_k]
    fs = []
    beta = beta0
    counter = 0
    is_converged = True

    # time_per_iteration = []
    while gap_k > eps:
        # start_per_iter = time.perf_counter()
        counter += 1
        if verbose:
            print(f"Newton step: {counter}")
        f_k = f(beta, X, y)
        grad_k = grad(beta, X, y)
        hessian_k = hessian(beta, X, y)
        nu_k = -np.linalg.solve(hessian_k, grad_k)
        t = 1
        if backtracking:
            bt_counter = 0
            lhs = f(beta + t * nu_k, X, y)
            rhs = f_k + 0.5 * t * np.dot(grad_k, nu_k)
            while lhs > rhs:
                bt_counter += 1
                t = bt_factor * t
                lhs = f(beta + t * nu_k, X, y)
                rhs = f_k + 0.5 * t * np.dot(grad_k, nu_k)
            if verbose:
                print(f"Number of backtracking steps: {bt_counter}")
            beta = beta + t * nu_k
            gap_k = gap(beta, X, y)
            gaps.append(gap_k)
            fs.append(f_k)
            if verbose:
                print(f"Current f: {f_k}")
                print(f"Current gap: {gap_k}")
            if counter > 50:
                is_converged = False
                break
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
    return fs, gaps, is_converged, beta
