import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

eurusd = pd.read_csv("pytorch\\data\\20230605-tsk.hl.bse.6.21.txt",
                     names=["Open","High","Low","Close","tal","kal","sal","tas","kas","sas"])

train = eurusd.copy()
tal = train.pop("tal")
kal = train.pop("kal")
sal = train.pop("sal")
tas = train.pop("tas")
kas = train.pop("kas")
sas = train.pop("sas")

targets = pd.DataFrame(tal)
targets = targets.merge(kal,left_index=True, right_index=True)
targets = targets.merge(sal,left_index=True, right_index=True)
targets = targets.merge(tas,left_index=True, right_index=True)
targets = targets.merge(kas,left_index=True, right_index=True)
targets = targets.merge(sas,left_index=True, right_index=True)

normalize = layers.Normalization()
normalize.adapt(train)

model = keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(6)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())

model.fit(train, targets, epochs=20)