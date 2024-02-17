import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.signal import chirp
import pandas as pd

torch.random.manual_seed(0)

data = pd.read_csv(
    "pytorch\\data\\EUR.USD-Day-Trade.txt",
    header=0,
)
signal = data.values[:,0]
w = signal.reshape(1,signal.shape[0])


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

signal = torch.FloatTensor(w)
print(signal.shape)

# Define transform
spectrogram = T.Spectrogram(n_fft=40, win_length=20)

# Perform transform
spec = spectrogram(signal)
print(spec[0].shape)

fig, axs = plt.subplots(2, 1)
plot_waveform(signal, 1, title="Original waveform", ax=axs[0])
plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.show()
