import glob
import pandas as pd

files = glob.glob("pytorch\\data\\*.csv");
df_list = (pd.read_csv(file,names=["id","open","high","low","close","range","m.open","m.high","m.low","m.close","m.range","tal","sal","kal","tas","sas","kas","lookback"]) for file in files)
df   = pd.concat(df_list, ignore_index=True)

df = df.loc[:,["open","high","low","close","range","m.open","m.high","m.low","m.close","m.range","tal"]]
df.to_csv("pytorch\\data\\tal.csv",index=False,header=True)