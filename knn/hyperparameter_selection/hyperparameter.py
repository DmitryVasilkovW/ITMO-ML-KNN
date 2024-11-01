from sklearn.model_selection import KFold, cross_val_score
from knn.impl.knn_impl import KNearestNeighbors
from knn.separation.axis_repo import DataRepo


class KNNModelSelector:
    best_model = None
    best_params = None
    best_score = 0
    param_grid = {
        'metric': ['euclidean', 'manhattan', 'cosine'],
        'kernel': ['uniform', 'gaussian', 'generalized', 'exponential'],
        'base_n_neighbors': [3, 5, 9],
        'dynamic_window': [True, False]
    }

    @classmethod
    def _set_params(cls, param_grid, attribute, cv_folds=5):
        x_train = DataRepo.get_axis("x", "train", attribute)
        y_train = DataRepo.get_axis("y", "train", attribute)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cls._check_for_metric(param_grid, x_train, y_train, kf)

    @classmethod
    def _check_for_metric(cls, param_grid, x_train, y_train, kf):
        for metric in param_grid['metric']:
            cls._check_for_kernels(param_grid, metric, x_train, y_train, kf)

    @classmethod
    def _check_for_kernels(cls, param_grid, metric, x_train, y_train, kf):
        for kernel in param_grid['kernel']:
            cls._check_for_base_n_neighbors(param_grid, metric, kernel, x_train, y_train, kf)

    @classmethod
    def _check_for_base_n_neighbors(cls, param_grid, metric, kernel, x_train, y_train, kf):
        for base_n_neighbors in param_grid['base_n_neighbors']:
            cls._check_for_dynamic_window(param_grid, metric, kernel, base_n_neighbors, x_train, y_train, kf)

    @classmethod
    def _check_for_dynamic_window(cls, param_grid, metric, kernel, base_n_neighbors, x_train, y_train, kf):
        for dynamic_window in param_grid['dynamic_window']:
            cls._try_to_update(metric, kernel, base_n_neighbors, dynamic_window, x_train, y_train, kf)

    @classmethod
    def _try_to_update(cls, metric, kernel, base_n_neighbors, dynamic_window, x_train, y_train, kf):
        __knn = KNearestNeighbors(
            base_n_neighbors=base_n_neighbors,
            kernel=kernel,
            metric=metric,
            dynamic_window=dynamic_window
        )

        scores = cross_val_score(__knn, x_train, y_train, cv=kf, scoring='accuracy')
        mean_accuracy = scores.mean()

        if mean_accuracy > cls.best_score:
            cls.best_score = mean_accuracy
            cls.best_params = {
                'metric': metric,
                'kernel': kernel,
                'base_n_neighbors': base_n_neighbors,
                'dynamic_window': dynamic_window
            }
            cls.best_model = __knn

    @classmethod
    def get_best_model(cls, param_grid=None, attribute="Class"):
        if cls.best_model is None and param_grid is None:
            cls._set_params(cls.param_grid, attribute)
        elif param_grid is not None:
            cls._set_params(param_grid, attribute)
        return cls.best_model

    @classmethod
    def get_best_params(cls, param_grid=None, attribute="Class"):
        if cls.best_params is None and param_grid is None:
            cls._set_params(cls.param_grid, attribute)
        elif param_grid is not None:
            cls._set_params(param_grid, attribute)
        return cls.best_params

    @classmethod
    def get_best_score(cls, param_grid=None, attribute="Class"):
        if cls.best_score is None and param_grid is None:
            cls._set_params(cls.param_grid, attribute)
        elif param_grid is not None:
            cls._set_params(param_grid, attribute)
        return cls.best_score
