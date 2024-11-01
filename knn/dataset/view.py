import pandas as pd
from sklearn.preprocessing import LabelEncoder

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"


def get_data():
    data = pd.read_csv(url, header=None)

    data.columns = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]

    data.replace({'x': 1, 'o': 0, 'b': -1}, inplace=True)

    label_encoder = LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])

    return data
