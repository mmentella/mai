import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import tqdm
import copy

# read data
data = pd.read_csv("pytorch\\data\\tal.csv", header=0)
X = data.iloc[:, 0:25]
y = data.iloc[:, 25]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=1)
ada = AdaBoostClassifier(
    estimator=tree, n_estimators=400, learning_rate=0.1, random_state=1
)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')