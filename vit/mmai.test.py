import pandas as pd
import torch

# read csv
data = pd.read_csv(
    "pytorch\\data\\mmai.vit.channels.features-1Days-EUR.USD-RAW.csv",
    header=None,
)

# exclude index
data = data.iloc[:, 1:]

# prepare features and label
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

X = torch.FloatTensor(X)
print(X.type(), X.shape)

# build channels
X = X.reshape(X.shape[0],4,64)
print(X.type(), X.shape)
print(X)

# build sequences
X = X.reshape(X.shape[0],4,16,4)
print(X.type(), X.shape)
print(X[1,1,:,:])