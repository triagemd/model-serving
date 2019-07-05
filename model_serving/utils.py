import numpy as np
from sklearn.preprocessing import normalize


def l2_normalize(x, copy=False, return_norm=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        return normalize(x.reshape(1, -1), copy=copy, return_norm=return_norm)
    else:
        return normalize(x, copy=copy, return_norm=return_norm)
