import numpy as np

def subvariance(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance =  np.sum((x - x_mean) * (y - y_mean)) / len(x)
    return covariance /(np.std(x) * np.std(y))

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X
    corr = np.zeros((X.shape[1], Y.shape[1]))

    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            corr[i, j] = subvariance(X[:, i], Y[:, j])

    return corr
		