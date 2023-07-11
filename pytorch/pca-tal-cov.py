import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pytorch\\data\\tal.csv",header=None)
even = df[::2]
even = even.reset_index(drop=True)
odd = df[1::2]
odd = odd.reset_index(drop=True)

df8=pd.concat([even,odd],axis=1)
print(df8)