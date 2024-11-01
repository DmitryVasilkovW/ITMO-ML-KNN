from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from knn.hyperparameter_selection.hyperparameter import KNNModelSelector
from knn.impl.knn_impl import KNearestNeighbors
from knn.lowess.lowess import show_difference_between_weighted_and_unweighted
from knn.separation.axis_repo import DataRepo


def run(from_val, to_val):
    neighbors_range = range(from_val, to_val)
    train_scores = []
    test_scores = []

    repo = DataRepo()
    x_train = repo.get_axis("x", "train", "Class")
    y_train = repo.get_axis("y", "train", "Class")
    x_test = repo.get_axis("x", "test", "Class")
    y_test = repo.get_axis("y", "test", "Class")

    params = KNNModelSelector()
    best_params = params.get_best_params()

    for n_neighbors in neighbors_range:
        _knn = KNearestNeighbors(base_n_neighbors=n_neighbors, kernel=best_params['kernel'],
                                 metric=best_params['metric'],
                                 dynamic_window=best_params['dynamic_window'])
        _knn.fit(x_train, y_train)

        y_pred_train = _knn.predict(x_train)
        y_pred_test = _knn.predict(x_test)

        train_mae = accuracy_score(y_train, y_pred_train)
        test_mae = accuracy_score(y_test, y_pred_test)

        train_scores.append(train_mae)
        test_scores.append(test_mae)

    return neighbors_range, train_scores, best_params, test_scores


def show_plot():
    neighbors_range, train_scores, _, test_scores = run(1, 25)

    plt.plot(neighbors_range, train_scores, label='Train accuracy', marker='o')
    plt.plot(neighbors_range, test_scores, label='Test accuracy', marker='o')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Neighbors')
    plt.legend()
    plt.show()


def show_difference():
    show_difference_between_weighted_and_unweighted("Class")


def show_best_params():
    _, _, best_params, _ = run(1, 25)

    print(best_params)
