import numpy as np


def grad_L1_elementwise(beta_i: float):
    """
    Calculates the gradient of the L1 norm for a single element
    :param beta_i:
    :return:
    """
    if beta_i == 0:
        return 2 * np.random.random(1) - 1
    return np.sign(beta_i)


def grad_L1(beta: np.ndarray):
    """
    Calculates the gradient of the L1 norm for an array.
    :param beta:
    :return:
    """
    return np.vectorize(grad_L1_elementwise)(beta)


def soft_thresholding(theta: np.ndarray, threshold: float):
    """
    Calculates the soft thresholding function for an array.
    :param theta:
    :param threshold:
    :return:
    """
    def soft_thresholding_i(theta_i, threshold):
        if theta_i > threshold:
            return theta_i - threshold
        elif theta_i < -threshold:
            return theta_i + threshold
        return 0

    return np.vectorize(soft_thresholding_i)(theta, threshold)
