"""Collection of functions to prepare data for further analysis."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_test_train(x1, x2, y, train_fraction=0.8):
    """Split data into train and test datasets."""
    # Use scikit learn to split data
    in_train, in_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, train_size=train_fraction)

    # Split in_train and in_test in x1 and x2 again
    x1_train = in_train[:, 0:x1.shape[0]]
    x2_train = in_train[:, x1.shape[0]:]

    x1_test = in_test[:, 0:x1.shape[0]]
    x2_test = in_test[:, x1.shape[0]:]

    return x1_train, x2_train, x1_test, x2_test, y_train, y_test


def bootstrap(x1, x2, y):
    """Perform boostrap method to data."""
    data = np.vstack([x1.flatten(), x2.flatten(), y.flatten()]).T

    data_size = data.shape[0]

    data_boot = np.empty([data.shape[0], data.shape[1]])

    for row in range(data_size):
        random_index = int(np.random.uniform(0, data_size, 1))
        data_boot[row, :] = data[random_index, :]

    x1_boot = data_boot[:, 0].reshape(x1.shape[0], x1.shape[1])
    x2_boot = data_boot[:, 1].reshape(x2.shape[0], x2.shape[1])
    y_boot = data_boot[:, 2].reshape(y.shape[0], y.shape[1])

    return x1_boot, x2_boot, y_boot