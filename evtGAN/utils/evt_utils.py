from scipy.stats import genextreme
import numpy as np

def fit_gev(train_data):
    """Fit GEV to each location in the training data."""
    n_train, h, w = train_data.shape
    params = np.zeros((h, w, 3))  # [c, loc, scale]
    for i in range(h):
        for j in range(w):
            data = train_data[:, i, j]
            c, loc, scale = genextreme.fit(data)
            params[i, j] = [c, loc, scale]
    return params

def transform_back(u_gen, params):
    """Transform uniform samples back to original scale using GEV."""
    n_gen, c, h, w = u_gen.shape  # u_gen: [n_gen, 1, h, w]
    z_gen = np.zeros((n_gen, h, w))
    for i in range(h):
        for j in range(w):
            c_val, loc, scale = params[i, j]
            u = u_gen[:, 0, i, j]  # Extract channel 0
            z_gen[:, i, j] = genextreme.ppf(u, c_val, loc=loc, scale=scale)
    return z_gen