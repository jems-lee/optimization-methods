import numpy as np
import src.utils as utils


def neg_loglike(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the negative log likelihood for logistic regression.

    :param beta: parameters
    :param X: covariates
    :param y: response
    :return: value
    """
    value = 0
    for i in range(len(y)):
        XTbeta = np.dot(X[i, :], beta)
        value += np.log(1 + np.exp(XTbeta)) - y[i] * XTbeta
    return value


def neg_loglike_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Calculates the negative gradient of the log likelihood for logistic regression.

    :param beta: parameters
    :param X: covariates
    :param y: response
    :return:
    """
    value = np.zeros(len(beta))
    for i in range(len(y)):
        xi = X[i, :]
        expxibeta = np.exp(np.dot(xi, beta))
        value += xi * expxibeta / (1 + expxibeta) - y[i] * xi
    return value


def neg_loglike_hessian(beta, X, y):
    """
    Calculates the negative hessian of the log likelihood for logistic regression.

    :param beta:
    :param X:
    :param y:
    :return:
    """
    p = len(beta)
    value = np.zeros((p, p))
    for i in range(len(y)):
        xi = X[i, :]
        expxibeta = np.exp(np.dot(xi, beta))
        value += -np.outer(xi, xi) * expxibeta / (1 + expxibeta) ** 2
    return value


def diag_hessian(beta, X, y):
    """
    Calculates the negative diagonal of the hessian matrix

    :param beta:
    :param X:
    :param y:
    :return:
    """
    n = len(y)
    p = len(beta)
    B = np.zeros((p, p))
    for i in range(n):
        xi = X[i, :]
        # logcoeff = np.dot(xi, beta) - 2*np.log(1 + np.exp( np.dot(xi, beta) ))
        # coef = np.exp(logcoeff)
        coef = np.exp(np.dot(xi, beta)) / (1 + np.exp(np.dot(xi, beta))) ** 2
        B += -coef * np.diag(xi ** 2)

    # return np.maximum(np.diag(B), 0)
    return np.diag(B)


def logisticgap(beta, X, y):
    grad = neg_loglike_grad(beta, X, y)
    hessian = neg_loglike_hessian(beta, X, y)
    gap = 0.5 * np.dot(grad, np.linalg.solve(hessian, grad))
    return gap


def opt_f(beta, X, y, reg):
    return neg_loglike(beta, X, y) + reg * np.linalg.norm(beta, 1)


def opt_f_grad(beta, X, y, reg):
    return neg_loglike_grad(beta, X, y) + reg * utils.grad_L1(beta)

