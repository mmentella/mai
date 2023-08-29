import glob
import pandas as pd

files = glob.glob("pytorch\\data\\mmai.transformers.features.auto-15Minute-EUR.USD-RAW.csv")
df_list = (
    pd.read_csv(
        file,
        index_col=0,
        names=[
            "id",
            "open",
            "high",
            "low",
            "close",
            "label"
        ],
    )
    for file in files
)
df = pd.concat(df_list, ignore_index=True)

df = df.loc[
    :,
    [
        "open",
        "high",
        "low",
        "close",
        "label"
    ],
]

df.to_csv("pytorch\\data\\mmai.transformers.features.csv", index=False, header=True)
