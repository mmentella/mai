import torch

from model.encoder import Encoder
from model.embedding import Embedding


class GatedTransformer(torch.nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_hidden: int,
        dim_feature: int,
        dim_timestep: int,
        dim_q: int,
        dim_v: int,
        num_heads: int,
        num_encoders: int,
        num_class: int,
        dropout: float = 0.2,
    ):
        super(GatedTransformer, self).__init__()

        self.timestep_embedding = Embedding(
            dim_feature=dim_feature,
            dim_timestep=dim_timestep,
            dim_model=dim_model,
            wise="timestep",
            pos_mode='static'
        )
        self.feature_embedding = Embedding(
            dim_feature=dim_feature,
            dim_timestep=dim_timestep,
            dim_model=dim_model,
            wise="feature",
            pos_mode='static'
        )

        self.timestep_encoderlist = torch.nn.ModuleList(
            [
                Encoder(
                    dim_model=dim_model,
                    dim_hidden=dim_hidden,
                    dim_q=dim_q,
                    dim_v=dim_v,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoders)
            ]
        )

        self.feature_encoderlist = torch.nn.ModuleList(
            [
                Encoder(
                    dim_model=dim_model,
                    dim_hidden=dim_hidden,
                    dim_q=dim_q,
                    dim_v=dim_v,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoders)
            ]
        )

        self.gate = torch.nn.Linear(
            in_features=dim_timestep * dim_model + dim_feature * dim_model,
            out_features=2,
        )
        self.linear_out = torch.nn.Linear(
            in_features=dim_timestep * dim_model + dim_feature * dim_model,
            out_features=num_class,
        )

    def forward(self, x: torch.Tensor, stage: str = "train" or "test"):
        x_timestep, _ = self.timestep_embedding(x)
        x_feature, _ = self.feature_embedding(x)

        for encoder in self.timestep_encoderlist:
            x_timestep, heatmap = encoder(x_timestep, stage=stage)

        for encoder in self.feature_encoderlist:
            x_feature, heatmap = encoder(x_feature, stage=stage)

        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
        x_feature = x_feature.reshape(x_feature.shape[0], -1)

        gate = torch.nn.functional.softmax(
            self.gate(torch.cat([x_timestep, x_feature], dim=-1)), dim=-1
        )

        gate_out = torch.cat(
            [x_timestep * gate[:, 0:1], x_feature * gate[:, 1:2]], dim=-1
        )

        out = self.linear_out(gate_out)

        return out
