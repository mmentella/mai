import torch
import torch.nn as nn
import pandas as pd
from einops import rearrange, repeat

x = torch.arange(1,5).reshape((2,2))

mask = torch.ones_like(x[0])
print(mask)
mask = mask.tril(diagonal=0)
print(mask)