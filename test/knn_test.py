import unittest

import numpy as np
import pandas as pd

from knn.impl.knn_impl import KNearestNeighbors
from knn.impl.service.kernel_function import uniform_function, \
    gaussian_function


class TestKNearestNeighbors(unittest.TestCase):

    def setUp(self):
        self.knn = KNearestNeighbors(base_n_neighbors=3, weights='uniform', kernel='uniform', metric='euclidean')
        self.x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_train = np.array([0, 1, 0, 1])
        self.knn.fit(self.x_train, self.y_train)

    def test_fit(self):
        np.testing.assert_array_equal(self.knn.x_train, self.x_train)
        np.testing.assert_array_equal(self.knn.y_train, self.y_train)

    def test_predict_uniform_weights(self):
        test_points = np.array([[1, 2]])
        prediction = self.knn.predict(test_points)
        self.assertEqual(prediction[0], 0)

    def test_kernel_uniform(self):
        self.assertEqual(self.knn._add_kernel(0), uniform_function())

    def test_kernel_gaussian(self):
        self.knn.kernel = 'gaussian'
        distance = 1
        self.assertEqual(self.knn._add_kernel(distance), gaussian_function(distance))

    def test_dynamic_window_size(self):
        self.knn.dynamic_window = True
        distances = [0.5, 1.0, 1.5]
        window_size = self.knn._get_window_size(distances)
        self.assertEqual(window_size, len(distances))

    def test_set_params(self):
        params = {'base_n_neighbors': 4, 'kernel': 'gaussian'}
        self.knn.set_params(**params)
        self.assertEqual(self.knn.base_n_neighbors, 4)
        self.assertEqual(self.knn.kernel, 'gaussian')

    def test_set_metric(self):
        self.knn.set_metric('chebyshev')
        self.assertEqual(self.knn.model.metric, 'chebyshev')

    def test_sample_weights(self):
        sample_weights = np.array([1, 1, 0.5, 2])
        self.knn.fit(self.x_train, self.y_train, sample_weights=sample_weights)
        test_points = np.array([[3, 4]])
        prediction = self.knn.predict(test_points)
        self.assertIn(prediction[0], [0, 1])

    def test_predict_with_series_labels(self):
        y_train_series = pd.Series(self.y_train)
        self.knn.fit(self.x_train, y_train_series)
        test_points = np.array([[2, 3]])
        prediction = self.knn.predict(test_points)
        self.assertIn(prediction[0], [0, 1])


if __name__ == '__main__':
    unittest.main()
