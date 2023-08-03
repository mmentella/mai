import torch
import torch.nn as nn

c = nn.Conv1d(1, 16, 5, 1, 2)
b = nn.BatchNorm1d(16)
p = nn.MaxPool1d(kernel_size=2)

l = nn.Conv1d(16, 32, 5, 1, 2)
m = nn.BatchNorm1d(32)
n = nn.MaxPool1d(kernel_size=2)

rnd = torch.rand(16, 1, 24)

x = c(rnd)
y = b(x)
z = p(y)

print(f"input {rnd.shape}")
print(f"conv1 {x.shape}")
print(f"batch {y.shape}")
print(f"pool1 {z.shape}")

x = l(z)
y = m(x)
z = n(y)

print(f"conv2 {x.shape}")
print(f"batch {y.shape}")
print(f"pool2 {z.shape}")

z = z.view(z.size(0), -1)
print(f"flatten {z.shape}")

fc = nn.Linear(192, 3)
out = fc(z)
print(f'out {out}')