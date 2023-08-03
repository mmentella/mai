import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

df = pd.read_csv("pytorch\\data\\buysell.csv")
print(df.head())

# Class Distribution
sns.countplot(x="label", data=df)
#plt.show()

# Encode Output Class
class2idx = {"FLAT": 0, "LONG": 1, "SHORT": 2}

idx2class = {v: k for k, v in class2idx.items()}

df["label"].replace(class2idx, inplace=True)
print(df.head())

# Create Input and Output Data
X = df.iloc[:, 0:24]
y = df.iloc[:, 24]

# Train — Validation — Test
# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=69
)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21
)

# Normalize Input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


# Visualize Class Distribution in Train, Val, and Test
def get_class_distribution(obj):
    count_dict = {
        "flat": 0,
        "long": 0,
        "short": 0,
    }

    for i in obj:
        if i == 0:
            count_dict["flat"] += 1
        elif i == 1:
            count_dict["long"] += 1
        elif i == 2:
            count_dict["short"] += 1
        else:
            print("Check classes.")

    return count_dict


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
# Train
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[0],
).set_title("Class Distribution in Train Set")
# Validation
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[1],
).set_title("Class Distribution in Val Set")
# Test
sns.barplot(
    data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(),
    x="variable",
    y="value",
    hue="variable",
    ax=axes[2],
).set_title("Class Distribution in Test Set")
#plt.show()


# Custom Dataset
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset = ClassifierDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
)
val_dataset = ClassifierDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
)
test_dataset = ClassifierDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
)

# weighted sampling
target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
print(class_weights)

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all, num_samples=len(class_weights_all), replacement=True
)

# model parameters
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 3

train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


# define neural network architecture
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        # Constraints for layer 1
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.batch1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )  # default stride is equivalent to the kernel_size

        # Constraints for layer 2
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.batch2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Defining the Linear layer
        in_features = int(
            32 * num_feature / 4
        )  # 4 = due layer max pool con kernel size 2
        self.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        # Max Pool 1
        out = self.pool1(out)
        
        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        
        # Max Pool 2
        out = self.pool2(out)
        
        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fc(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


# train the model
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


accuracy_stats = {"train": [], "val": []}
loss_stats = {"train": [], "val": []}

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
        X_train_batch = torch.unsqueeze(X_train_batch,1)
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

            y_val_pred = model(torch.unsqueeze(X_val_batch,1))

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats["train"].append(train_epoch_loss / len(train_loader))
    loss_stats["val"].append(val_epoch_loss / len(val_loader))
    accuracy_stats["train"].append(train_epoch_acc / len(train_loader))
    accuracy_stats["val"].append(val_epoch_acc / len(val_loader))

    print(
        f" Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}"
    )

# visualize loss and accuracy
# Create dataframes
train_val_acc_df = (
    pd.DataFrame.from_dict(accuracy_stats)
    .reset_index()
    .melt(id_vars=["index"])
    .rename(columns={"index": "epochs"})
)
train_val_loss_df = (
    pd.DataFrame.from_dict(loss_stats)
    .reset_index()
    .melt(id_vars=["index"])
    .rename(columns={"index": "epochs"})
)
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
sns.lineplot(
    data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]
).set_title("Train-Val Accuracy/Epoch")
sns.lineplot(
    data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]
).set_title("Train-Val Loss/Epoch")
plt.show()

# test the model
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = torch.unsqueeze(X_batch.to(device),1)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(
    columns=idx2class, index=idx2class
)

sns.heatmap(confusion_matrix_df, annot=True)
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_list))
