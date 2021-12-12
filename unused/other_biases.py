def findBias2(y_train, y_test, y_trainfit, y_testfit):
    y = np.vstack([y_train, y_test])
    y_fit = np.vstack([y_trainfit, y_testfit])
    y_mean = np.mean(y_fit)
    return np.mean((y - y_mean) ** 2)


def findBias3(y_train, y_test, y_trainfit, y_testfit):
    y = np.vstack([y_train, y_test])
    y_fit = np.vstack([y_trainfit, y_testfit])
    y_mean = np.mean(y_fit)
    return np.mean((y - y_mean))


def findBias4(y_data, y_fit):
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_fit), 1)
    y_mean = np.mean(y_fit)
    return np.mean((y_data - y_mean))