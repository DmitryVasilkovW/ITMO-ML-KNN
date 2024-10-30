import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from knn.dataset import view
from knn.separation.select_data import DataProcessor

data = view.get_data()
processor = DataProcessor(data)
x_train = processor.get("x", "train")
y_train = processor.get("y", "train")
x_test = processor.get("x", "test")

knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

knn_cosine = KNeighborsRegressor(n_neighbors=5, weights='uniform', metric='cosine')
knn_cosine.fit(x_train, y_train)


def custom_kernel(distances, a=1, b=2):
    return (1 - np.abs(distances) ** a) ** b


knn_custom = KNeighborsRegressor(n_neighbors=5, weights=custom_kernel, metric='minkowski', p=2)
knn_custom.fit(x_train, y_train)
