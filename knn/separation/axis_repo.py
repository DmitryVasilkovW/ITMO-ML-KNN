from knn.dataset.view import get_data
from knn.separation.select_data import DataProcessor


class DataRepo:
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    data = None
    repo = None

    @classmethod
    def __get_x(cls, data_type: str):
        assert cls.repo is not None and isinstance(cls.repo, DataProcessor)

        if data_type == 'train':
            if cls.x_train is None:
                cls.x_train = cls.repo.get("x", data_type)
            return cls.x_train

        if data_type == 'test':
            if cls.x_test is None:
                cls.x_test = cls.repo.get("x", data_type)
            return cls.x_test

        return None

    @classmethod
    def __get_y(cls, data_type: str):
        assert cls.repo is not None and isinstance(cls.repo, DataProcessor)

        if data_type == 'train':
            if cls.y_train is None:
                cls.y_train = cls.repo.get("y", data_type)
            return cls.y_train

        if data_type == 'test':
            if cls.y_test is None:
                cls.y_test = cls.repo.get("y", data_type)
            return cls.y_test

        return None

    @classmethod
    def get_axis(cls, axis: str, data_type: str, attribute="Class"):
        if cls.data is None:
            cls.data = get_data()
        if cls.repo is None:
            cls.repo = DataProcessor(data=cls.data, attribute=attribute)

        if axis == 'x':
            return cls.__get_x(data_type)
        if axis == 'y':
            return cls.__get_y(data_type)

        return None
