import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

from knn.hyperparameter_selection.hyperparameter import KNNModelSelector
from knn.impl.knn_impl import KNN
from knn.separation.axis_repo import DataRepo


def compute_fitted_value(X, y, weights, fraction, i):
    """Вычисляет значение для одной точки."""
    distances = np.linalg.norm(X - X[i], axis=1)
    bandwidth = np.percentile(distances, 100 * fraction)

    local_weights = (1 - (distances / bandwidth) ** 3) ** 3 * weights
    W = np.diag(local_weights)

    b = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y.values)
    return X[i] @ b


def lowess(X, y, fraction=0.25, num_iterations=3):
    n = len(y)
    fitted_values = np.zeros(n)
    weights = np.ones(n)

    for iteration in range(num_iterations):
        fitted_values = Parallel(n_jobs=-1)(
            delayed(compute_fitted_value)(X, y, weights, fraction, i) for i in range(n)
        )

        residuals = y.values - fitted_values
        mad = np.median(np.abs(residuals))

        weights = 1 / (1 + (residuals / mad) ** 2)

    return fitted_values


def show_difference_between_weighted_and_unweighted():
    repo = DataRepo()
    X_train = repo.get_axis("x", "train")
    y_train = repo.get_axis("y", "train")
    X_test = repo.get_axis("x", "test")
    y_test = repo.get_axis("y", "test")

    params = KNNModelSelector()
    best_params = params.get_best_params()

    fitted_train = lowess(X_train, y_train)

    # Определение весов на основе остатков
    residuals_train = y_train - fitted_train
    weights_train = np.clip(residuals_train / np.std(residuals_train), 0, None)

    knn = KNN(base_n_neighbors=15, kernel=best_params['kernel'], metric=best_params['metric'],
              dynamic_window=best_params['dynamic_window'])
    knn.fit(X_train, y_train)

    pred_test_no_weights = knn.predict(X_test)

    knn.fit(X_train, y_train, sample_weights=weights_train)
    pred_test_with_weights = knn.predict(X_test)

    mae_no_weights_test = mean_absolute_error(y_test, pred_test_no_weights)
    mae_with_weights_test = mean_absolute_error(y_test, pred_test_with_weights)

    print("MAE без весов (тестовый набор):", mae_no_weights_test)
    print("MAE с весами (тестовый набор):", mae_with_weights_test)
