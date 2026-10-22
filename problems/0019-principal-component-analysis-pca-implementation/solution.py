import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    x_c = (data - data.mean(axis=0))/ data.std(axis=0)
    c = (x_c.T @ x_c)/(data.shape[0] - 1)
    values, vectors = np.linalg.eigh(c)
    w = vectors[:, -1:-k-1:-1]
    non_zero = w[np.argmax(np.abs(w) > 0, axis=0), np.arange(k)]
    sign = np.where(non_zero > 0, 1.0, -1.0)

    return np.round(w * sign, 4)
