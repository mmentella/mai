import torch
import torch.nn as nn

from einops import rearrange, repeat

x = torch.randn(1,64)
print(x.shape)
x = rearrange(x, 'b (s p) -> b s p', s = 16,p=4)
print(x.shape)

x = torch.randn(1,1,4)
x = repeat(x, '1 1 d -> b 1 d', b = 16)
print(x.shape)