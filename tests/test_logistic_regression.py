import pytest
import numpy as np
from src.logistic_regression import (
    neg_loglike,
    neg_loglike_grad,
    neg_loglike_hessian,
    diag_hessian,
)


def generate_random_X_y(n, p):
    X = np.random.random((n, p))
    y = np.random.random(n)
    return X, y


def generate_same(n, p, fun=np.zeros):
    X = fun((n, p))
    y = fun(n)
    beta = fun(p)
    return beta, X, y


class TestLogistic:
    def test_zero_scalar(self):
        beta, X, y = generate_same(1, 1)
        assert neg_loglike(beta, X, y) == np.log(2)

    def test_zero_2d(self):
        beta, X, y = generate_same(2, 2)
        assert neg_loglike(beta, X, y) == 2 * np.log(2)

    def test_one_scalar(self):
        beta, X, y = generate_same(1, 1, np.ones)
        assert neg_loglike(beta, X, y) == (np.log(1 + np.exp(1)) - 1)

    def test_one_nd(self):
        n = 10
        p = 5
        beta, X, y = generate_same(n, p, np.ones)
        assert neg_loglike(beta, X, y) == n * (np.log(1 + np.exp(p)) - p)


class TestGradient:
    def test_zero_1d(self):
        beta, X, y = generate_same(1, 1)
        assert neg_loglike_grad(beta, X, y) == 0

    def test_zero_2d(self):
        beta, X, y = generate_same(2, 2)
        assert np.allclose(neg_loglike_grad(beta, X, y), np.zeros(2))

    def test_X_is_ones_2d(self):
        beta, X, y = generate_same(1, 1, np.ones)
        assert neg_loglike_grad(beta, X, y) == np.array([(np.exp(1) / (1 + np.exp(1)) - 1)])


class TestHessian:
    def test_zero_scalar(self):
        beta, X, y = generate_same(1, 1)
        assert neg_loglike_hessian(beta, X, y) == np.array([[0]])

    def test_zero_2d(self):
        beta, X, y = generate_same(2, 2)
        assert np.allclose(neg_loglike_hessian(beta, X, y), np.zeros((2,2)))

    def test_one_scalar(self):
        beta, X, y = generate_same(1, 1, np.ones)
        assert np.allclose(neg_loglike_hessian(beta, X, y), -np.exp(1) / (1 + np.exp(1))**2 * np.ones(1))

    def test_one_2d(self):
        beta, X, y = generate_same(2, 2, np.ones)
        assert np.allclose(
            neg_loglike_hessian(beta, X, y), -2 * np.exp(2) / (1 + np.exp(2))**2 * np.outer(np.ones(2), np.ones(2))
        )

    def test_beta_zeros_vs_diag(self):
        X, y = generate_random_X_y(n=5, p=5)
        beta = np.zeros(X.shape[1])
        assert np.allclose(
            np.diag(neg_loglike_hessian(beta, X, y)), diag_hessian(beta, X, y)
        )

    def test_beta_normal_vs_diag(self):
        X, y = generate_random_X_y(n=5, p=5)
        beta = np.zeros(X.shape[1])
        assert np.allclose(
            np.diag(neg_loglike_hessian(beta, X, y)), diag_hessian(beta, X, y)
        )

    def test_beta_normal_large_vs_diag(self):
        X, y = generate_random_X_y(n=50, p=100)
        beta = np.zeros(X.shape[1])
        assert np.allclose(
            np.diag(neg_loglike_hessian(beta, X, y)), diag_hessian(beta, X, y)
        )
