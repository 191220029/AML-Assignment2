import numpy as np

def get_data(path: str):
    data = np.loadtxt(path, dtype=float, delimiter=',')
    return data
