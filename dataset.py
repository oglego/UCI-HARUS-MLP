import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UCIHAR(Dataset):
    def __init__(self, root, train=True):
        split = "train" if train else "test"
        X_path = f"{root}/{split}/X_{split}.txt"
        y_path = f"{root}/{split}/y_{split}.txt"

        self.X = pd.read_csv(X_path, delim_whitespace=True, header=None).values.astype(np.float32)
        self.y = pd.read_csv(y_path, delim_whitespace=True, header=None).values.flatten().astype(np.int64) - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
