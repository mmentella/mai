import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(
        self,
        dim_feature: int,
        dim_timestep: int,
        dim_model: int,
        wise: str = "timestep" or "feature",
        pos_mode: str = "static" or "learnable",
    ):
        super(Embedding, self).__init__()

        self.dim_feature = dim_feature
        self.dim_timestap = dim_timestep
        self.dim_model = dim_model
        self.wise = wise
        self.pos_mode = pos_mode

        if wise == "timestep":
            self.embedding = torch.nn.Linear(dim_feature, dim_model)
        elif wise == "feature":
            self.embedding = torch.nn.Linear(dim_timestep, dim_model)

    def forward(self, x: torch.Tensor):
        if self.wise == "feature":
            x = self.embedding(x)
        elif self.wise == "timestep":
            x = self.embedding(x.transpose(-1, -2))
            x = (
                position_encode_static(x)
                if self.pos_mode == "static"
                else position_encode_learnable(x)
            )

        return x, None


def position_encode_static(x: torch.Tensor):
    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)

    return x + pe


def position_encode_learnable(x: torch.Tensor):
    pe = nn.Parameter(torch.randn(1, x.shape[-2], x.shape[-1]))

    return x + pe
