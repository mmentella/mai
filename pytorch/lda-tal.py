import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from plot import plot_decision_regions

df = pd.read_csv("pytorch\\data\\tal.csv", header=None)
X, y = df.iloc[:, 0:4].values, df.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
labels = dict.fromkeys(y_train)
i = 1
for l in labels:
    labels[l] = i
    i+=1
l_train = np.empty_like(y_train)
for idx,y in enumerate(y_train):
    l_train[idx] = labels[y]
X_train_lda = lda.fit_transform(X_train_std, l_train)

lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")
lr.fit(X_train_lda, l_train)

plot_decision_regions(X_train_lda, l_train, classifier=lr)
plt.tight_layout()
plt.show()
