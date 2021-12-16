"""Collection of functions to prepare data for further analysis."""
import numpy as np
from sklearn.model_selection import train_test_split


def scale_data(data):
    """Use min-max normalization to scale data."""
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data


def split_test_train(x1, x2, y, train_fraction=0.8, do_shuffle=True):
    """Split data into train and test datasets."""
    # Use scikit learn to split data
    in_train, in_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, train_size=train_fraction,
                                                          shuffle=do_shuffle)

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


def cross_validation(x1, x2, y, num_fold):
    """Perform cross-validation to data."""
    # Reshape input
    x1 = x1.flatten().reshape(x1.shape[0]*x1.shape[1], 1)
    x2 = x2.flatten().reshape(x2.shape[0]*x2.shape[1], 1)
    y = y.flatten().reshape(y.shape[0]*y.shape[1], 1)

    x = np.hstack([x1, x2])

    # Define number of points per fold
    num_points = np.shape(x)[0]
    num_points_fold = int(np.floor(num_points/num_fold))

    # Shuffle data
    ind = np.arange(num_points)
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]

    # Initialize placeholders
    x_test_arr = np.zeros([num_fold, num_points_fold, 2])
    y_test_arr = np.zeros([num_fold, num_points_fold])

    x_train_arr = np.zeros([num_fold, num_points-num_points_fold, 2])
    y_train_arr = np.zeros([num_fold, num_points-num_points_fold])

    # Split data
    for fold_index in np.arange(num_fold):
        test_ind = np.arange(fold_index*num_points_fold, (fold_index+1)*num_points_fold)
        train_ind = np.delete(np.arange(num_points), test_ind)

        x_test_arr[fold_index, :, :] = x[test_ind]
        y_test_arr[fold_index, :] = y[test_ind, 0]
        x_train_arr[fold_index, :, :] = x[train_ind]
        y_train_arr[fold_index, :] = y[train_ind, 0]

    return x_train_arr, x_test_arr, y_train_arr, y_test_arr