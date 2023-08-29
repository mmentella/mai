from dataloader import EURUSD, EURUSDFLAT
from fx import FXT, Deep

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)

dataloader = DataLoader(EURUSD(), 32, shuffle=False)
model = FXT(n_features=4, sequence_len=16, embedding_dim=64, n_heads=8, n_blocks=10)
print(sum([x.reshape(-1).shape[0] for x in model.parameters()]))
# exit()
N_EPOCHS = 20
LR = 0.005

optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        y = y.reshape(y.shape[0], 1)
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(dataloader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
