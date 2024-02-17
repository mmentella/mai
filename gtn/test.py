from model.embedding import Embedding
import torch

# embedding = Embedding(4,16,64,'feature',pos_mode='static')
x=torch.randn(2,64)
print(x)

x = x.reshape(x.shape[0],16,4)
print(x)

x  = x.transpose(-1,-2)
print(x)