import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
import joblib
import utils


def read_test_series_from_file(file_name, train_ratio=utils.train_ratio):
    fin = open(file_name, 'r')

    data = fin.read().splitlines()
    series = list(map(float, data))

    series = series[int(len(series) * train_ratio):]

    fin.close()
    return series


def get_w(model):
    w = [value for value in model.coef_[0]]
    return w


model_file_name = 'model.bin'
model = joblib.load(model_file_name)

test_series = read_test_series_from_file('data.txt')
X_test, y_test = utils.build_data(test_series)

y_predict = model.predict(X_test)


mse = mean_squared_error(y_test, y_predict)
rmse = math.sqrt(mse)
print('Root mean squared error:', rmse)


utils.visualize_data(y_test, y_predict)
