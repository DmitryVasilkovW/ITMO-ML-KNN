import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from knn.dataset import view
from knn.separation.select_data import DataProcessor

data = view.get_data()
processor = DataProcessor(data, "Heating_Load")
neighbors = range(1, 20)
train_errors = []
test_errors = []
x_train = processor.get("x", "train")
y_train = processor.get("y", "train")
x_test = processor.get("x", "test")
y_test = processor.get("y", "test")

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    train_errors.append(knn.score(x_train, y_train))
    test_errors.append(knn.score(x_test, y_test))

plt.plot(neighbors, train_errors, label="Training score")
plt.plot(neighbors, test_errors, label="Test score")
plt.xlabel("Number of Neighbors")
plt.ylabel("R-squared")
plt.legend()
plt.show()
