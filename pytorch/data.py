import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

df = pd.read_csv("pytorch\\data\\buysell.csv")

# Encode Output Class
class2idx = {"FLAT": 0, "ENTRY": 1}

idx2class = {v: k for k, v in class2idx.items()}

df["label"].replace(class2idx, inplace=True)
# print(df.head())

# sns.countplot(x = 'label', data=df)
# plt.show()

# Create Input and Output Data
X = df.iloc[:, 0:24]
y = df.iloc[:, 24]

print(y)

sm = SMOTE(random_state=12)
X_smote, y_smote = sm.fit_resample(X, y)

print(y_smote)