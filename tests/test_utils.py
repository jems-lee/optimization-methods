import numpy as np
import pytest

from src.utils import soft_thresholding, grad_L1_elementwise, grad_L1


class TestSoftThresholding:
    def test_one_large_positive_element(self):
        assert soft_thresholding(theta=1, threshold=0.5) == 0.5

    def test_one_large_negative_element(self):
        assert soft_thresholding(theta=-1, threshold=0.5) == -0.5

    def test_one_small_positive_and_one_small_negative_element(self):
        assert (
            soft_thresholding(theta=np.array([-0.25, 0.25]), threshold=0.5)
            == np.zeros(2)
        ).all()

    def test_with_all_cases(self):
        theta = np.array([-1, -0.5, 0.25, 1])
        expected = np.array([-0.5, 0, 0, 0.5])
        assert (soft_thresholding(theta=theta, threshold=0.5) == expected).all()


class TestGradL1:
    def test_positive(self):
        assert grad_L1_elementwise(5) == 1

    def test_negative(self):
        assert grad_L1_elementwise(-6) == -1

    def test_zero(self):
        assert -1 < grad_L1_elementwise(0) < 1

    def test_all_nonzero_cases(self):
        theta = np.array([-5, -0.5, 1, 5])
        expected = np.array([-1, -1, 1, 1])
        assert (grad_L1(theta) == expected).all()

    def test_multiple_zeros(self):
        theta = np.array([0, 0, 0])
        assert (-1 < grad_L1(theta)).all() and (grad_L1(theta) < 1).all()

