from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from knn.dataset import view
from knn.separation import select_data

data = view.get_data()
processor = select_data.DataProcessor(data)

param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'metric': ['minkowski', 'cosine'],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(processor.get("x", "train"), processor.get("y", "train"))

print(f"Best parameters: {grid_search.best_params_}")
