import pandas as pd

data = pd.read_csv("pytorch\\data\\buysell.csv", header=0)
data_clean = data.drop_duplicates(
    subset=[
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
        "F13",
        "F14",
        "F15",
        "F16",
        "F17",
        "F18",
        "F19",
        "F20",
        "F21",
        "F22",
        "F23",
        "F24",
    ],
    keep='last',
)

data_clean.to_csv("pytorch\\data\\buysell.csv", index=False, header=True)