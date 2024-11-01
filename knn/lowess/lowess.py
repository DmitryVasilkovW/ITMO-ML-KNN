import numpy as np
from sklearn.metrics import accuracy_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from knn.hyperparameter_selection.hyperparameter import KNNModelSelector
from knn.impl.knn_impl import KNN
from knn.separation.axis_repo import DataRepo


def lowess_smoothing(x, y, fraction=0.25):
    if x.ndim > 1:
        x = x[:, 0]

    smoothed_values = lowess(y, x, frac=fraction, return_sorted=False)
    return smoothed_values


def show_difference_between_weighted_and_unweighted(attribute="Heating_Load"):
    x_train = DataRepo.get_axis("x", "train", attribute)
    y_train = DataRepo.get_axis("y", "train", attribute)
    x_test = DataRepo.get_axis("x", "test", attribute)
    y_test = DataRepo.get_axis("y", "test", attribute)

    params = KNNModelSelector()
    best_params = params.get_best_params()

    fitted_train = lowess_smoothing(x_train, y_train)

    residuals_train = y_train - fitted_train
    weights_train = np.clip(residuals_train / np.std(residuals_train), 0, None)

    knn = KNN(base_n_neighbors=best_params['base_n_neighbors'],
              kernel=best_params['kernel'],
              metric=best_params['metric'],
              dynamic_window=best_params['dynamic_window'])
    knn.fit(x_train, y_train)

    pred_test_no_weights = knn.predict(x_test)

    knn.fit(x_train, y_train, sample_weights=weights_train)
    pred_test_with_weights = knn.predict(x_test)

    mae_no_weights_test = accuracy_score(y_test, pred_test_no_weights)
    mae_with_weights_test = accuracy_score(y_test, pred_test_with_weights)

    print("MAE без весов (тестовый набор):", mae_no_weights_test)
    print("MAE с весами (тестовый набор):", mae_with_weights_test)
