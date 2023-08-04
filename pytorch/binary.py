import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# read data
data = pd.read_csv("pytorch\\data\\buysell.csv", header=0)
X = data.iloc[:, 0:24]
y = data.iloc[:, 24]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

scaler = StandardScaler()

X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

hidden_size_wide = 128
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(24, hidden_size_wide)
        self.act = nn.Tanh()
        self.output = nn.Linear(hidden_size_wide, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

hidden_size_deep = 32
class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(24, hidden_size_deep)
        self.act1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size_deep, hidden_size_deep)
        self.act2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_size_deep, hidden_size_deep)
        self.act3 = nn.Tanh()
        self.output = nn.Linear(hidden_size_deep, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


# Compare model sizes
model1 = Wide()
model2 = Deep()
print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041


# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 100  # number of epochs to run
    batch_size = 16  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores_wide = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Wide()
    acc = model_train(
        model, X_train[train], y_train[train], X_train[test], y_train[test]
    )
    print("Accuracy (wide): %.2f" % acc)
    cv_scores_wide.append(acc)
cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Deep()
    acc = model_train(
        model, X_train[train], y_train[train], X_train[test], y_train[test]
    )
    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)

# evaluate the model
wide_acc = np.mean(cv_scores_wide)
wide_std = np.std(cv_scores_wide)
deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)
print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc * 100, wide_std * 100))
print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc * 100, deep_std * 100))

# rebuild model with full set of training data
if wide_acc > deep_acc:
    print("Retrain a wide model")
    model = Wide()
else:
    print("Retrain a deep model")
    model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        y_pred = model(X_test[i : i + 1])
        print(
            f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})"
        )

    # Plot the ROC curve
    y_pred = model(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)  # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
