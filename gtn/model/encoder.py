import torch
from model.multiHeadAttention import MultiHeadAttention
from model.feedforward import PositionFeedforward


class Encoder(torch.nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_v: int,
        num_heads: int,
        dim_model: int,
        dim_hidden: int,
        dropout: float = 0.2,
    ):
        super(Encoder, self).__init__()

        self.mha = MultiHeadAttention(
            dim_model=dim_model, dim_q=dim_q, dim_v=dim_v, num_heads=num_heads
        )
        self.ff = PositionFeedforward(
            dim_model=dim_model, dim_hidden=dim_hidden
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layernorm = torch.nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, stage: str):
        residual = x
        
        x, heatmap_score = self.mha(x, stage)
        x = self.dropout(x)
        x = self.layernorm(x + residual)

        x = self.ff(x)

        return x, heatmap_score
