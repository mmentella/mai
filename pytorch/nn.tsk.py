import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TskNet(nn.Module):
    def __init__(self):
        super(TskNet, self).__init__()
        self.inputLayer = nn.Linear(16, 256, True)
        self.outputLayer = nn.Linear(256, 6, True)

    def forward(self, x):
        x = self.inputLayer(x);
        x = F.tanh(x)
        x = self.outputLayer(x)
        x = F.sigmoid(x)
        return x

tsknet = TskNet()
print(tsknet)

params = list(tsknet.parameters())
print(len(params))
print(params[0].size())

input = torch.tensor([1.08985,1.08990,1.08985,1.08990,1.08990,1.08995,1.08990,1.08995,
                      1.08995,1.08995,1.08985,1.08985,1.08985,1.08990,1.08985,1.08990])
out = tsknet(input)
print(out)

target = torch.tensor([0.05,0.05,0.05,0.0125,0.0125,0.0125])
target = target.view(6)
criterion = nn.MSELoss()

loss = criterion(out, target)

tsknet.zero_grad()
print(tsknet.inputLayer.bias.grad)

loss.backward()
print(tsknet.inputLayer.bias.grad)

# create your optimizer
optimizer = optim.SGD(tsknet.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = tsknet(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update