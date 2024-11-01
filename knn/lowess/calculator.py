import numpy as np

const_weight = 3
percent = 100


def _calculate_distances(x, i):
    return np.linalg.norm(x - x[i], axis=1)


def _calculate_bandwidth(distances, fraction):
    return np.percentile(distances, percent * fraction)


def _calculate_local_weights(distances, bandwidth, weights):
    local_weights = (1 - (distances / bandwidth) ** const_weight) ** const_weight * weights
    return np.diag(local_weights)


def _calculate_coefficients(x, weight, y):
    return np.linalg.pinv(x.T @ weight @ x) @ (x.T @ weight @ y.values)


def get_lowess(x, y, weights, fraction, i):
    distances = _calculate_distances(x, i)
    bandwidth = _calculate_bandwidth(distances, fraction)
    weight = _calculate_local_weights(distances, bandwidth, weights)
    b = _calculate_coefficients(x, weight, y)
    return x[i] @ b
