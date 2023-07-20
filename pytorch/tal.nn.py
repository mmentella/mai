import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import tqdm

# read data
data = pd.read_csv("pytorch\\data\\tal.csv", header=0)

X = data.iloc[:, 0:10]
Y = data.iloc[:, 10:]

for idx, c in enumerate(np.unique(Y.values)):
    Y.replace(c, idx, inplace=True)

X = X.values
Y = Y.values

x, x_val, y, y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

x_train = x.reshape(-1, x.shape[1]).astype("float32")
y_train = y.reshape(-1)

x_val = x_val.reshape(-1, x_val.shape[1]).astype("float32")
y_val = y_val.reshape(-1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)


class Data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(x_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


data_set = Data()
trainloader = DataLoader(dataset=data_set, batch_size=64)


class Net(nn.Module):
    def __init__(self, num_feature, num_class):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


input_dim = 10  # how many Variables are in the dataset
output_dim = 13  # number of classes

model = Net(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 1000
accuracy_stats = {"train": [], "val": []}
loss_stats = {"train": [], "val": []}

# n_epochs
for epoch in tqdm(range(1, n_epochs + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0

    model.train()
    for x, y in trainloader:
        # clear gradient
        optimizer.zero_grad()
        # make a prediction
        z = model(x)
        # calculate loss
        loss = criterion(z, y)
        acc = multi_acc(z, y)
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        train_epoch_loss += loss.item()
        train_epoch_acc += acc.item()

        # print("epoch {}, loss {}".format(epoch, loss.item()))

z = model(x_val)
yhat = torch.max(z.data, 1)

print(yhat)
print(y_val)
