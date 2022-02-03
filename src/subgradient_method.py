import numpy as np

def subgradient_descent_logistic(beta, X, y, f, subgrad, reg, max_iter=200, t=None):
    f_values = np.zeros(max_iter)
    best_f_values = np.zeros(max_iter)
    best_iters = 0
    min_f = np.inf
    best_beta = beta
    for iter in range(max_iter):
        step_size = t if t is not None else (1 / (iter + 1))
        grad_step = subgrad(beta=beta, X=X, y=y, reg=reg)
        beta = beta - step_size * grad_step
        f_value = f(beta=beta, X=X, y=y, reg=reg)
        f_values[iter] = f_value
        if f < min_f:
            best_f_values[best_iters] = f
            min_f = f
            best_beta = beta
            best_iters += 1
    return f_values, best_f_values[:best_iters], best_beta