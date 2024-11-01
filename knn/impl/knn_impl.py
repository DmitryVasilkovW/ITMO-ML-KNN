import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNearestNeighbors:
    def __init__(self, base_n_neighbors=5, weights='uniform', kernel='uniform', metric='euclidean', min_neighbors=1,
                 dynamic_window=False):
        self.y_train = None
        self.x_train = None
        self.base_n_neighbors = base_n_neighbors
        self.weights = weights
        self.kernel = kernel
        self.metric = metric
        self.min_neighbors = min_neighbors
        self.dynamic_window = dynamic_window
        self.sample_weights = None
        self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=self.metric)

    def fit(self, x, y, sample_weights=None):
        self.x_train = x
        self.y_train = y
        if sample_weights is not None:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = np.ones(len(x))

        self.sample_weights = sample_weights if sample_weights is not None else np.ones(len(x))
        self.model.fit(x)

    def predict(self, X):
        """Предсказание меток классов для новых данных."""
        predictions = []
        for point in X:
            distances, indices = self.model.kneighbors([point], n_neighbors=self.base_n_neighbors)
            n_neighbors = self._get_window_size(distances[0])
            distances, indices = self.model.kneighbors([point], n_neighbors=n_neighbors)
            if self.weights == 'uniform':
                predictions.append(self._predict_uniform(indices))
            elif self.weights == 'distance':
                predictions.append(self._predict_distance(distances, indices))
        return np.array(predictions)

    def _get_window_size(self, distances):
        """Определение размера окна в зависимости от выбранной стратегии."""
        if self.dynamic_window:
            return max(self.min_neighbors, len(distances))
        else:
            return self.base_n_neighbors

    def _predict_uniform(self, indices):
        weighted_votes = {}
        neighbors = indices[0]
        for neighbor in neighbors:
            if isinstance(self.y_train, pd.Series):
                label = self.y_train.iloc[neighbor]
            else:
                label = self.y_train[neighbor]

            if self.sample_weights is not None:
                if isinstance(self.sample_weights, pd.Series):
                    weight = self.sample_weights.iloc[neighbor]
                else:
                    weight = self.sample_weights[neighbor]
            else:
                weight = 1

            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        return max(weighted_votes, key=weighted_votes.get)

    def _predict_distance(self, distances, indices):
        """Предсказание с использованием взвешенного голосования по расстоянию с учётом априорных весов."""
        predictions = []
        for i, neighbors in enumerate(indices):
            weighted_votes = {}
            for j, neighbor in enumerate(neighbors):
                distance = distances[i][j]
                kernel_weight = self._apply_kernel(distance)
                label = self.y_train.iloc[neighbor]
                weight = kernel_weight * self.sample_weights[neighbor]
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
            predictions.append(max(weighted_votes, key=weighted_votes.get))
        return predictions

    def _apply_kernel(self, distance):
        """Применение выбранного ядра для вычисления веса."""
        if self.kernel == 'uniform':
            return 1
        elif self.kernel == 'gaussian':
            return np.exp(-distance ** 2)
        elif self.kernel == 'generalized':
            return (1 - distance) ** 2
        elif self.kernel == 'exponential':
            return np.exp(-distance)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def set_metric(self, metric, p=None):
        """Установка метрики для вычисления расстояний."""
        if metric == 'minkowski' and p is not None:
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=metric, p=p)
        elif metric == 'chebyshev':
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric='chebyshev')
        else:
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=metric)

    def set_window(self, n_neighbors):
        """Установка базового размера окна (количества соседей)."""
        self.base_n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=self.metric)

    def get_params(self, deep=True):
        return {
            'base_n_neighbors': self.base_n_neighbors,
            'kernel': self.kernel,
            'metric': self.metric,
            'dynamic_window': self.dynamic_window
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
