import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class EURUSD(Dataset):
    def __init__(self) -> None:
        data = pd.read_csv(
            "pytorch\\data\\mmai.transformers.features.auto-15Minute-EUR.USD-RAW.csv",
            header=0,
        )
        data = data.iloc[:, 1:]

        X = data.iloc[:, 0:64].values
        y = data.iloc[:, 64].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X = X.reshape(X.shape[0], 16, 4)
        y = y.reshape(y.shape[0], 1)

        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class EURUSDFLAT(Dataset):
    def __init__(self) -> None:
        data = pd.read_csv(
            "pytorch\\data\\mmai.transformers.features.auto-15Minute-EUR.USD-RAW.csv",
            header=0,
        )
        data = data.iloc[:, 1:]

        X = data.iloc[:, 0:64].values
        y = data.iloc[:, 64].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # X = X.reshape(X.shape[0], 16, 4)
        y = y.reshape(y.shape[0], 1)

        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]