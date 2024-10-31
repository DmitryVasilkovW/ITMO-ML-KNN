from sklearn.model_selection import KFold, cross_val_score
from knn.impl.knn_impl import KNN
from knn.separation.axis_repo import DataRepo


class KNNModelSelector:
    best_model = None
    best_params = None
    best_score = float('inf')
    param_grid = {
        'metric': ['euclidean', 'manhattan', 'cosine'],
        'kernel': ['uniform', 'gaussian', 'generalized', 'exponential'],
        'base_n_neighbors': [3, 5, 8],
        'dynamic_window': [True, False]
    }

    @classmethod
    def _grid_search_knn(cls, param_grid, cv_folds=5):
        X_train = DataRepo.get_axis("x", "train")
        y_train = DataRepo.get_axis("y", "train")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for metric in param_grid['metric']:
            for kernel in param_grid['kernel']:
                for base_n_neighbors in param_grid['base_n_neighbors']:
                    for dynamic_window in param_grid['dynamic_window']:
                        __knn = KNN(
                            base_n_neighbors=base_n_neighbors,
                            kernel=kernel,
                            metric=metric,
                            dynamic_window=dynamic_window
                        )

                        scores = cross_val_score(__knn, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
                        mean_mae = -scores.mean()

                        if mean_mae < cls.best_score:
                            cls.best_score = mean_mae
                            cls.best_params = {
                                'metric': metric,
                                'kernel': kernel,
                                'base_n_neighbors': base_n_neighbors,
                                'dynamic_window': dynamic_window
                            }
                            cls.best_model = __knn

    @classmethod
    def get_best_model(cls, param_grid=None):
        if cls.best_model is None and param_grid is None:
            cls._grid_search_knn(cls.param_grid)
        elif cls.best_model is None and param_grid is not None:
            cls._grid_search_knn(param_grid)
        return cls.best_model

    @classmethod
    def get_best_params(cls, param_grid=None):
        if cls.best_params is None and param_grid is None:
            cls._grid_search_knn(cls.param_grid)
        elif cls.best_params is None and param_grid is not None:
            cls._grid_search_knn(param_grid)
        return cls.best_params

    @classmethod
    def get_best_score(cls, param_grid=None):
        if cls.best_score is None and param_grid is None:
            cls._grid_search_knn(cls.param_grid)
        elif cls.best_score is None and param_grid is not None:
            cls._grid_search_knn(param_grid)
        return cls.best_score
