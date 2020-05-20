import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def visualize_data(x_data, y_data, x_line=None, y_line=None):
    plt.plot(x_data, y_data, 'ro')

    if x_line is not None and y_line is not None:
        plt.plot(x_line, y_line)

    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()


def solve_by_formula(x_mat, y):
    # Building x_mat_bar
    one = np.ones((x_mat.shape[0], 1))
    x_mat_bar = np.concatenate((one, x_mat), axis=1)

    # Calculating weights of the fitting line
    a_mat_dagger = np.dot(x_mat_bar.T, x_mat_bar)
    b = np.dot(x_mat_bar.T, y)
    w = np.dot(np.linalg.pinv(a_mat_dagger), b)
    print('w = ', w.T)

    # Preparing the fitting line
    w_0 = w[0][0]
    w_1 = w[1][0]
    x_line = np.linspace(145, 185, 2)
    y_line = w_0 + w_1 * x_line

    # Drawing the fitting line, data = (x_mat, y), the fitting line = (x_line, y_line)
    visualize_data(x_mat, y, x_line, y_line)


def solve_by_sklearn(x_mat, y):
    # Building x_mat_bar
    one = np.ones((x_mat.shape[0], 1))
    x_mat_bar = np.concatenate((one, x_mat), axis=1)

    # fit the model by Linear Regression
    # fit_intercept = False for calculating the bias
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x_mat_bar, y)

    print('w = ', regr.coef_)


# height (cm)
x_mat = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Visualize data
visualize_data(x_mat, y)


solve_by_formula(x_mat, y)

solve_by_sklearn(x_mat, y)
