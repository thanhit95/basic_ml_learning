import numpy as np
from sklearn.metrics import mean_squared_error
# from math import sqrt


y_expected = [0.0, 0.5, 0.0, 0.5, 0.0]
y_expected = np.array(y_expected).reshape(len(y_expected), 1)

y_predicted = [0.15, 0.5, 0.0, 0.3, 0.2]
y_predicted = np.array(y_predicted).reshape(len(y_predicted), 1)


# mean forecast error
mfe = (y_predicted - y_expected).mean()

# mean absolute error
mae = (np.abs(y_predicted - y_expected)).mean()

# mean squared error
mse = (np.square(y_predicted - y_expected)).mean()
# mse = mean_squared_error(y_predicted, y_expected)

# root mean squared error
rmse = np.sqrt(mse)
print(rmse)
