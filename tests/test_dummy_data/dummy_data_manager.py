import numpy as np


def load_dummy_data(len_x_train, len_y_train, len_x_test, len_y_test, num_data_sources=1, low=0, high=100):
    x_train = [np.random.randint(low=low, high=high, size=(len_x_train, 1)) for _ in range(num_data_sources)]
    y_train = np.random.randint(low=low, high=high, size=(len_y_train, 1))
    x_test = [np.random.randint(low=low, high=high, size=(len_x_test, 1)) for _ in range(num_data_sources)]
    y_test = np.random.randint(low=low, high=high, size=(len_y_test, 1))

    return x_train, y_train, x_test, y_test
