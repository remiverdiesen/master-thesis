import numpy as np
import pandas as pd
import torch

def load_and_preprocess_data(data_file, ids_file):
    # Load data
    df = pd.read_csv(data_file, sep=',', header=None).iloc[1:].values.astype(float).T
    ids_EU = pd.read_csv(ids_file, sep=',', header=None).iloc[1:].values.astype(int) - 1

    # Transform to uniform using ECDF
    def ecdf(data):
        n = len(data)
        ranks = np.apply_along_axis(lambda x: np.argsort(x) + 1, 0, data)
        return ranks / (n + 1)

    df = ecdf(df)

    # Reshape and pad
    n_lat, n_lon = len(np.unique(ids_EU[:, 1])), len(np.unique(ids_EU[:, 0]))
    df = df.reshape(-1, n_lat, n_lon, 1)
    df = np.pad(df, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    return torch.tensor(df, dtype=torch.float32), torch.tensor(ids_EU, dtype=torch.long)

def get_relevant_points(data, ids):
    batch_size = data.size(0)
    num_points = ids.size(0)
    ids_lat = ids[:, 1].view(1, -1).expand(batch_size, -1)
    ids_lon = ids[:, 0].view(1, -1).expand(batch_size, -1)
    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_points)
    return data[batch_indices, ids_lat, ids_lon, 0]  # [batch_size, num_points]