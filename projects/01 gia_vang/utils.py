import numpy as np
import matplotlib.pyplot as plt


train_ratio = 0.7


def build_data(series, group_prev_elements=5):
    x_series = []
    y_series = []

    for i in range(group_prev_elements, len(series)):
        x_vector = [series[val_idx]
                    for val_idx in range(i - group_prev_elements, i)]
        x_vector.insert(0, 1)
        x_series.extend(x_vector)
        y_series.append(series[i])

    X_bar = np.array(x_series).reshape(len(y_series), group_prev_elements + 1)
    y = np.array(y_series).reshape(len(y_series), 1)

    return X_bar, y


def visualize_data(y, y_predict=None):
    x = list(range(len(y)))
    plt.figure(figsize=(15, 8))

    plt.plot(x, y, 'g-', markersize=2)

    if y_predict is not None:
        plt.plot(x, y_predict, 'r-', markersize=2)

    plt.axis([0, len(y), 20, 60])
    plt.xlabel('Ngày')
    plt.ylabel('Giá')
    plt.title('GIÁ VÀNG')
    plt.show()
