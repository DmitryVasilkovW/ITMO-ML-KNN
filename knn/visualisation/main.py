from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from knn.hyperparameter_selection.hyperparameter import KNNModelSelector
from knn.impl.knn_impl import KNearestNeighbors
from knn.lowess.lowess import show_difference_between_weighted_and_unweighted
from knn.separation.axis_repo import DataRepo

neighbors_range = range(1, 25)
train_scores = []
test_scores = []

repo = DataRepo()
X_train = repo.get_axis("x", "train", "Class")
y_train = repo.get_axis("y", "train", "Class")
X_test = repo.get_axis("x", "test", "Class")
y_test = repo.get_axis("y", "test", "Class")

params = KNNModelSelector()
best_params = params.get_best_params()

for n_neighbors in neighbors_range:
    _knn = KNearestNeighbors(base_n_neighbors=n_neighbors, kernel=best_params['kernel'], metric=best_params['metric'],
                             dynamic_window=best_params['dynamic_window'])
    _knn.fit(X_train, y_train)

    y_pred_train = _knn.predict(X_train)
    y_pred_test = _knn.predict(X_test)

    train_mae = accuracy_score(y_train, y_pred_train)
    test_mae = accuracy_score(y_test, y_pred_test)

    train_scores.append(train_mae)
    test_scores.append(test_mae)

plt.plot(neighbors_range, train_scores, label='Train MAE', marker='o')
plt.plot(neighbors_range, test_scores, label='Test MAE', marker='o')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE vs Number of Neighbors')
plt.legend()
plt.show()

print(best_params)
show_difference_between_weighted_and_unweighted("Class")
