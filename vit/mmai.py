import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f

from einops import repeat
from einops.layers.torch import Rearrange


class FxTransformer(nn.Module):
    def __init__(
        self,
        feature_in: int = 64,
        sequence_len: int = 16,
        patch_dim: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_encoders: int = 4,
        num_classes: int = 3,
        p_dropout: float = 0.5
    ):
        super().__init__()

        assert (
            feature_in == sequence_len * patch_dim
        ), "Features size feature_in doesn't match sequnce_len*patch_dim."

        # create patches from raw features
        # embed patches
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b (s p) -> b s p", s=sequence_len, p=patch_dim),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # positional encoding
        # we encode all patches - sequence_len - plus
        # the class token
        self.positional_encoding = nn.Parameter(
            torch.randn(1, sequence_len + 1, hidden_dim)
        )

        # transformer encoder
        self.transformer_encoder = FxTransformerEncoder(
            hidden_dim, num_heads, num_encoders
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(p_dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # feed model
        patches = self.to_patch_embedding(x)
        b, n, _ = patches.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        patches = torch.cat((cls_tokens, patches), dim=1)
        patches += self.positional_encoding[:, : (n + 1)]

        out = self.transformer_encoder(patches)

        cls = out[:, 0]
        return self.classifier(cls)


class FxTransformerEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, num_encoders: int = 4):
        super().__init__()

        # the encoder is composed of L identical layers.
        # Each one has two main subcomponents:
        #   (1) a multihead self-attention block (MSA)
        #   (2) a fully connected feed-forward dense block (MLP)
        self.layers = nn.Sequential()
        for _ in range(num_encoders):
            self.layers.append(FxTransformerLayer(dim, heads))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FxTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()

        # Each one has two main subcomponents:
        #   (1) a multihead self-attention block (MSA)
        #   (2) a fully connected feed-forward dense block (MLP)
        self.msa = FxMultiHeadAttention(dim, heads)
        # self.msa = MultiHeadAttention(dim, dim, dim, heads)
        self.mlp = FxFeedForward(dim)

        # layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        xnorm = self.norm(x)
        msa = self.msa(xnorm)

        msaresidual = msa + x
        # msaresidual = msa[0] + x

        msanorm = self.norm(msaresidual)

        mlp = self.mlp(msanorm)

        return mlp + msanorm


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_model: int, dim_q: int, dim_v: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_q = dim_q

        self.W_Q = torch.nn.Linear(
            in_features=dim_model, out_features=dim_q * num_heads
        )
        self.W_K = torch.nn.Linear(
            in_features=dim_model, out_features=dim_q * num_heads
        )
        self.W_V = torch.nn.Linear(
            in_features=dim_model, out_features=dim_v * num_heads
        )
        self.W_out = torch.nn.Linear(
            in_features=dim_v * num_heads, out_features=dim_model
        )

        self.inf = -(2**32) + 1
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x: torch.Tensor):
        Q = torch.cat(self.W_Q(x).chunk(self.num_heads, dim=-1), dim=0)
        K = torch.cat(self.W_K(x).chunk(self.num_heads, dim=-1), dim=0)
        V = torch.cat(self.W_V(x).chunk(self.num_heads, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) * (
            self.dim_q ** (-0.5)
        )  # / torch.sqrt(torch.Tensor(self.q)).to(self.device)

        heatmap_score = score

        score = torch.nn.functional.softmax(score, dim=-1)
        weight_V = torch.cat(
            torch.matmul(score, V).chunk(self.num_heads, dim=0), dim=-1
        )

        out = self.W_out(weight_V)

        return out, heatmap_score


class FxMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()

        self.dim = dim
        self.heads = heads

        self.attention = nn.ModuleList([FxHeadAttention(dim) for _ in range(heads)])
        self.linear = nn.Linear(heads * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(torch.cat([h(x) for h in self.attention], -1))


class FxHeadAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        qk = q.bmm(k.transpose(1, 2))
        scale = q.size(-1) ** 0.5
        softmax = f.softmax(qk / scale, dim=-1)

        return softmax.bmm(v)


class FxFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)
