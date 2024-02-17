import torch


class PositionFeedforward(torch.nn.Module):
    def __init__(self, dim_model: int, dim_hidden: int):
        super(PositionFeedforward, self).__init__()

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim_model, dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_hidden, dim_model),
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x):
        residual = x

        x = self.ff(x)
        x = self.layernorm(x + residual)

        return x
