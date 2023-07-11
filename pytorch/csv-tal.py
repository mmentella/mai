import glob
import pandas as pd

files = glob.glob("pytorch\\data\\*.csv");
df_list = (pd.read_csv(file,names=["id","open","high","low","close","tal","kal","sal","tas","kas","sas"]) for file in files)
df   = pd.concat(df_list, ignore_index=True)

df = df.loc[:,["open","high","low","close","tal"]]
df.to_csv("pytorch\\data\\tal.csv",index=False,header=False)