import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pytorch\\data\\tal.csv",header=None)
X,y = df.iloc[:,0:4].values,df.iloc[:,4].values

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1,5),var_exp,align='center', label='Individual explained variance')
# plt.step(range(1,5),cum_var_exp,where='mid', label='Cumulative explained variance')
# plt.ylabel('Explained Variance Ratio')
# plt.xlabel('Principal Component Index')
# plt.tight_layout()
# plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k:k[0],reverse=True)

W = np.hstack((eigen_pairs[0][1][:,np.newaxis],
               eigen_pairs[1][1][:,np.newaxis]))
X_train_pca = X_train_std.dot(W)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o','v','^','<','>','1','2','3','4','s']

for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],c=c,label=f'Class {l}',marker=m)
plt.tight_layout()
plt.show()