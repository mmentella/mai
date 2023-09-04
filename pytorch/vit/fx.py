import torch
import torch.nn as nn

import numpy as np


class FXT(nn.Module):
    def __init__(
        self,
        n_features: int = 4,
        sequence_len: int = 16,
        embedding_dim: int = 4,
        n_heads: int = 1,
        n_blocks: int = 1,
        out_d: int = 1,
    ):
        super(FXT, self).__init__()

        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(n_features, self.embedding_dim)
        self.class_token = nn.Parameter(torch.rand(1, self.embedding_dim))

        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(sequence_len + 1, self.embedding_dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [FxtBlock(self.embedding_dim, n_heads) for _ in range(n_blocks)]
        )

        self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, out_d), nn.Sigmoid())

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        b, s, f = src.shape
        src = self.embedding(src)
        src = torch.cat((self.class_token.expand(b, 1, -1), src), dim=1)

        out = src + self.positional_embeddings.repeat(b, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class FxtBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, mlp_ratio=4):
        super(FxtBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mhsa = FxtMSA(embedding_dim, n_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_ratio * embedding_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class FxtMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(FxtMSA, self).__init__()

        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class Deep(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = nn.Tanh()
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    

# define neural network architecture
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x