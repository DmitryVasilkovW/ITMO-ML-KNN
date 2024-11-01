import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from knn.impl.service.kernel_function import uniform_function, gaussian_function, generalized_function, \
    exponential_function


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

        self.model.fit(x)

    def predict(self, x):
        predictions = []
        for point in x:
            distances, indices = self.model.kneighbors([point], n_neighbors=self.base_n_neighbors)
            n_neighbors = self._get_window_size(distances[0])
            distances, indices = self.model.kneighbors([point], n_neighbors=n_neighbors)
            predictions = self._predictor(predictions, indices, distances)

        return np.array(predictions)

    def _predictor(self, predictions, indices, distances):
        if self.weights == 'uniform':
            predictions.append(self._receive_most_probable_uniform(indices))
        elif self.weights == 'distance':
            predictions.append(self._receive_most_probable_distance(distances, indices))
        return predictions

    def _get_window_size(self, distances):
        if self.dynamic_window:
            return max(self.min_neighbors, len(distances))
        else:
            return self.base_n_neighbors

    def _receive_most_probable_uniform(self, indices):
        weighted_votes = {}
        neighbors = indices[0]
        weighted_votes = self._check_all_neighbors_for_form(neighbors, weighted_votes)

        return max(weighted_votes, key=weighted_votes.get)

    def _check_all_neighbors_for_form(self, neighbors, weighted_votes):
        for neighbor in neighbors:
            label = self._update_label(neighbor)
            weight = self._receive_weights(neighbor)

            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        return weighted_votes

    def _update_label(self, neighbor):
        if isinstance(self.y_train, pd.Series):
            return self.y_train.iloc[neighbor]
        else:
            return self.y_train[neighbor]

    def _receive_weights(self, neighbor):
        if self.sample_weights is not None:
            return self._update_weights(neighbor)
        else:
            return 1

    def _update_weights(self, neighbor):
        if isinstance(self.sample_weights, pd.Series):
            return self.sample_weights.iloc[neighbor]
        else:
            return self.sample_weights[neighbor]

    def _receive_most_probable_distance(self, distances, indices):
        predictions = []
        predictions = self._check_all_indexes(distances, indices, predictions)
        return predictions

    def _check_all_indexes(self, distances, indices, predictions):
        for i, neighbors in enumerate(indices):
            weighted_votes = {}
            weighted_votes = self._check_all_neighbors_for_distance(distances, neighbors, weighted_votes, i)
            predictions.append(max(weighted_votes, key=weighted_votes.get))

        return predictions

    def _check_all_neighbors_for_distance(self, distances, neighbors, weighted_votes, ind):
        for j, neighbor in enumerate(neighbors):
            distance = distances[ind][j]
            kernel_weight = self._add_kernel(distance)
            label = self.y_train.iloc[neighbor]
            weight = kernel_weight * self.sample_weights[neighbor]
            if label in weighted_votes:
                weighted_votes[label] += weight
            else:
                weighted_votes[label] = weight

        return weighted_votes

    def _add_kernel(self, distance):
        if self.kernel == 'uniform':
            return uniform_function()
        elif self.kernel == 'gaussian':
            return gaussian_function(distance)
        elif self.kernel == 'generalized':
            return generalized_function(distance)
        elif self.kernel == 'exponential':
            return exponential_function(distance)
        else:
            raise ValueError(f"Incorrect kernel: {self.kernel}")

    def get_params(self, deep=True):
        return {
            'base_n_neighbors': self.base_n_neighbors,
            'kernel': self.kernel,
            'metric': self.metric,
            'dynamic_window': self.dynamic_window
        }

    def set_metric(self, metric, p=None):
        if metric == 'minkowski' and p is not None:
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=metric, p=p)
        elif metric == 'chebyshev':
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric='chebyshev')
        else:
            self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=metric)

    def set_window(self, n_neighbors):
        self.base_n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.base_n_neighbors, metric=self.metric)

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
