import pandas as pd

file_path = 'wine.data'

data = pd.read_csv(file_path, sep=',', header=None)

data_list = data.to_numpy().tolist()

attr = [row[1:] for row in data_list]  # cechy
cl = [row[0] for row in data_list]   # klasy

attributes = [[float(value) for value in row] for row in attr]
classes = [float(value) for value in cl]


def get_data():
    return attributes, classes
