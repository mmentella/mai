import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from umap import UMAP
from array import array

df = pd.read_csv("pytorch\\data\\tal.csv")

X, y = df.iloc[:, 0:10].values, df.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

reducer = UMAP(n_neighbors=25)
embedding = reducer.fit_transform(X_train_std)

labels = dict.fromkeys(y_train)
i = 0
for l in labels:
    labels[l] = i
    i += 1
l_train = array('i')
for idx, y in enumerate(y_train):
    l_train.append(int(labels[y]))

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[
        sns.color_palette()[x]
        for x in l_train

    ],
)
plt.gca().set_aspect("equal", "datalim")
plt.show()