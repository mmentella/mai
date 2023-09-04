import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pytorch\\data\\mmai.transformers.features-1Days-EUR.USD-RAW.csv")
X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 65), var_exp, align="center", label="Individual explained variance")
plt.step(range(1, 65), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Component Index")
plt.tight_layout()
plt.show()

eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
X_train_pca = X_train_std.dot(W)

# colors = ['tab:grey', 'tab:blue', 'tab:red']
colors = ['tab:blue', 'tab:red']
# markers = ["_", "^", "v"]
markers = ["^", "v"]

for l, c, m in zip(np.unique(y), colors, markers):
    plt.scatter(
        X_train_pca[y == l, 0],
        X_train_pca[y == l, 1],
        c=c,
        label=f"{l}",
        marker=m,
    )
plt.legend()
plt.tight_layout()
plt.show()
