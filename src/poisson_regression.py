import numpy as np
import time


def generate_data(p, loc_scale_factor, N, verbose=False):
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=N)
    betas = np.random.normal(loc=loc_scale_factor, scale=loc_scale_factor, size=p)
    mu = np.exp(np.sum(X * betas, axis=1))
    y = np.random.poisson(mu)
    beta0 = np.random.normal(loc=loc_scale_factor, scale=loc_scale_factor, size=p)
    if verbose:
        print(f"betas: {betas}")
        print(f"mu: {mu}")
        print(f"y: {y}")
        print(f"beta0: {beta0}")
    return X, betas, y, beta0


def poissonf(beta, X, y):
    value = 0
    for i in range(len(y)):
        XTbeta = np.dot(X[i, :], beta)
        value += np.exp(XTbeta) - y[i] * XTbeta
    return value


def poissongrad(beta, X, y):
    vector = np.zeros(len(beta))
    for i in range(len(y)):
        xi = X[i, :]
        vector += xi * np.exp(np.dot(xi, beta)) - y[i] * xi
    return vector


def poissonhessian(beta, X, y):
    p = len(beta)
    hessian = np.zeros((p, p))
    for i in range(len(y)):
        hessian += np.outer(X[i, :], X[i, :]) * np.exp(np.dot(X[i, :], beta))
    return hessian


def poissongap(beta, X, y):
    grad = poissongrad(beta, X, y)
    hessian = poissonhessian(beta, X, y)
    gap = 0.5 * np.dot(grad, np.linalg.solve(hessian, grad))
    return gap


def newtons_method(beta0, X, y, f, grad, hessian, gap, eps, backtracking, bt_factor, verbose=False):
    start = time.perf_counter()
    gap_k = gap(beta0, X, y)
    gaps = [gap_k]
    fs = []
    beta = beta0
    counter = 0
    is_converged = True
    time_per_iteration = []
    while gap_k > eps:
        start_per_iter = time.perf_counter()
        counter += 1
        if verbose:
            print(f"Newton step: {counter}")
        f_k = f(beta, X, y)
        grad_k = grad(beta, X, y)
        hessian_k = hessian(beta, X, y)
        nu_k = - np.linalg.solve(hessian_k, grad_k)
        t = 1
        if backtracking:
            bt_counter = 0
            lhs = f(beta + t * nu_k, X, y)
            rhs = f_k + .5 * t * np.dot(grad_k, nu_k)
            while lhs > rhs:
                bt_counter += 1
                t = bt_factor * t
                lhs = f(beta + t * nu_k, X, y)
                rhs = f_k + .5 * t * np.dot(grad_k, nu_k)
            if verbose:
                print(f"Number of backtracking steps: {bt_counter}")
            beta = beta + t * nu_k
            gap_k = gap(beta, X ,y)
            gaps.append(gap_k)
            fs.append(f_k)
            if verbose:
                print(f"Current f: {f_k}")
                print(f"Current gap: {gap_k}")
            if counter > 50:
                is_converged = False
                break
        end_per_iter = time.perf_counter()
        time_per_iteration.append(round(end_per_iter-start_per_iter,3))
    end = time.perf_counter()
    if is_converged:
        print(f"""Total iterations: {counter}
Total time: {end - start:.3f} seconds
Average time per iteration: {np.mean(time_per_iteration):.3f} seconds
        """)
    else:
        print("Convergence failed")
        print(f"Newton steps: {counter}")
    return fs, gaps, is_converged, beta