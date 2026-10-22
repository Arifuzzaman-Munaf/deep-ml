import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    # fit using mean and standard deviation
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    
    # Handling 0 standard deviation
    x_std = np.where(x_std == 0, 1, x_std)
    
    # Transform the test data
    return (X_test - x_mean) / x_std
