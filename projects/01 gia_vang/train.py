import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib
import utils


def read_series_from_file(file_name, train_ratio=utils.train_ratio):
    fin = open(file_name, 'r')

    data = fin.read().splitlines()
    series = list(map(float, data))

    series = series[: int(len(series) * train_ratio)]

    fin.close()
    return series


def solve_by_sklearn(X, y):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)

    print('w = ', model.coef_)
    return model


series = read_series_from_file('data.txt')
X_train, y_train = utils.build_data(series)

# utils.visualize_data(series)


model = solve_by_sklearn(X_train, y_train)


model_file_name = 'model.bin'
joblib.dump(model, model_file_name)
