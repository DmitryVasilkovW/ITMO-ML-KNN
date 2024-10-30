import statsmodels.api as sm
from matplotlib import pyplot as plt

from knn.dataset import view
from knn.separation.select_data import DataProcessor

data = view.get_data()
processor = DataProcessor(data)
x_train = processor.get("x", "train")
y_train = processor.get("y", "train")
x_test = processor.get("x", "test")
y_test = processor.get("y", "test")

lowess = sm.nonparametric.lowess
y_lowess = lowess(y_train, x_train[:, 0], frac=0.2)  # Используем первую характеристику для примера

# Визуализация
plt.scatter(x_train[:, 0], y_train, label="Original data")
plt.plot(y_lowess[:, 0], y_lowess[:, 1], color='red', label="LOWESS fit")
plt.xlabel("Relative Compactness")
plt.ylabel("Heating Load")
plt.legend()
plt.show()
