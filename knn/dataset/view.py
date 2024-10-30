import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
data = pd.read_excel(url)

data.columns = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
                'Overall_Height', 'Orientation', 'Glazing_Area',
                'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']

print(data.head())
print(data.describe())

print(data.isnull().sum())
