import torch
import numpy as np

def get_pos_coordinates(n_sub_ids):
    pos = [[i, j] for i in range(n_sub_ids) for j in range(i + 1, n_sub_ids)]
    return torch.tensor(pos, dtype=torch.long)

def compute_ECs(data, pos_coordinates):
    n = data.size(0)
    frechet = -torch.log(1 - data)  # To exponential
    ECs = []
    for pair in pos_coordinates:
        X1, X2 = frechet[:, pair[0]], frechet[:, pair[1]]
        minima = torch.min(X1, X2)
        EC = n / minima.sum()
        ECs.append(EC)
    return torch.stack(ECs)