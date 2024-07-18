import numpy as np

from test_dummy_data.dummy_data_manager import load_dummy_data


def test_load_dummy_data():
    x_train_len = 100
    y_train_len = 100
    x_test_len = 50
    y_test_len = 50
    num_data_sources = 3
    low = 0
    high = 100

    x_trains, y_train, x_tests, y_test = load_dummy_data(x_train_len, y_train_len, x_test_len, y_test_len,
                                                         num_data_sources, low, high)

    # Check lengths and shapes
    assert len(x_trains) == num_data_sources, "Number of x_train arrays is incorrect"
    assert all(x_train.shape == (x_train_len, 1) for x_train in x_trains), "x_train array shapes are incorrect"
    assert y_train.shape == (y_train_len, 1), "y_train shape is incorrect"

    assert len(x_tests) == num_data_sources, "Number of x_test arrays is incorrect"
    assert all(x_test.shape == (x_test_len, 1) for x_test in x_tests), "x_test shape is incorrect"
    assert y_test.shape == (y_test_len, 1), "y_test shape is incorrect"

    # Check data types
    assert all(isinstance(x_train, np.ndarray) for x_train in x_trains), "One or more x_train is not a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train is not a numpy array"
    assert all(isinstance(x_test, np.ndarray) for x_test in x_tests), "One or more x_test is not a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test is not a numpy array"

    # Check value ranges
    assert all(
        np.all((x_train >= low) & (x_train < high)) for x_train in x_trains), "x_train contains values out of range"
    assert np.all((y_train >= low) & (y_train < high)), "y_train contains values out of range"
    assert all(
        np.all((x_test >= low) & (x_test < high)) for x_test in x_tests), "x_test contains values out of range"
    assert np.all((y_test >= low) & (y_test < high)), "y_test contains values out of range"
