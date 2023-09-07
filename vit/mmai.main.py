import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

from mmai import FxTransformer

# read csv
data = pd.read_csv(
    "pytorch\\data\\mmai.transformers.features-1Days-EUR.USD-RAW.csv",
    header=0,
)
# exclude index
data = data.iloc[:, 1:]

# prepare features and label
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

print(X.shape)
print(y.shape)

# label encoder
labelEncoder = LabelEncoder()
labelEncoder.fit(y)
y = labelEncoder.transform(y)

decodeLabel = {k: v for k, v in enumerate(labelEncoder.classes_)}
print(decodeLabel)

# Train — Validation — Test
# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=69
)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21
)


# Custom Dataset
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# model parameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_dataset = ClassifierDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
)
val_dataset = ClassifierDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
)
test_dataset = ClassifierDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define moodel
model = FxTransformer(patch_dim=8,sequence_len=8,hidden_dim=128)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(sum([x.reshape(-1).shape[0] for x in model.parameters()]))


# train the model
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

###################################
print("Begin training.")
for e in tqdm(range(1, EPOCHS + 1), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
            device
        )
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    print(
        f" Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}"
    )

# test the model
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(
    columns=decodeLabel, index=decodeLabel
)

sns.heatmap(confusion_matrix_df, annot=True)
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_list))
