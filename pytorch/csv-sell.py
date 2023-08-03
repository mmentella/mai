import glob
import pandas as pd

files = glob.glob("pytorch\\data\\*SELL.csv")
df_list = (
    pd.read_csv(
        file,
        names=[
            "id",
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
            "F6",
            "F7",
            "F8",
            "F9",
            "F10",
            "F11",
            "F12",
            # "F13",
            # "F14",
            # "F15",
            # "F16",
            # "F17",
            # "F18",
            # "F19",
            # "F20",
            # "F21",
            # "F22",
            # "F23",
            # "F24",
            # "F25",
            "label",
        ],
    )
    for file in files
)
df = pd.concat(df_list, ignore_index=True)

df = df.loc[
    :,
    [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "F12",
        # "F13",
        # "F14",
        # "F15",
        # "F16",
        # "F17",
        # "F18",
        # "F19",
        # "F20",
        # "F21",
        # "F22",
        # "F23",
        # "F24",
        # "F25",
        "label",
    ],
]

df.to_csv("pytorch\\data\\sell.csv", index=False, header=True)
