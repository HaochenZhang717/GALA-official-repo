# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class ConditioningMLP(nn.Module):
    """
    Fusion module that integrates text embedding into the latent c representation.
    Supports 'add', 'concat', and 'gated_add' fusion modes.
    """

    def __init__(self, c_dim, text_dim, fusion_type='gated_add', pool_type='mean'):
        """
        Args:
            c_dim (int): The channel dimension of `c` (e.g., 64)
            text_dim (int): The dimension of text embeddings (e.g., 4096)
            fusion_type (str): One of ['add', 'concat', 'gated_add']
        """
        super().__init__()
        assert fusion_type in ['add', 'concat', 'gated_add'], \
            f"fusion_type must be one of ['add', 'concat', 'gated_add'], got {fusion_type}"
        assert pool_type in ['mean', 'max']

        self.fusion_type = fusion_type
        self.pool_type = pool_type
        self.c_dim = c_dim
        self.text_dim = text_dim

        # Text embedding MLP projection: text_dim -> c_dim
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, c_dim),
            nn.ReLU(inplace=True),
            nn.Linear(c_dim, c_dim)
        )

        # Optional gate for 'gated_add'
        if fusion_type == 'gated_add':
            self.gate_layer = nn.Sequential(
                nn.Linear(c_dim * 2, c_dim),
                nn.Sigmoid()
            )

    def forward(self, c, text_embedding):
        """
        Args:
            c (torch.Tensor): Conditioning input of shape [B, C, T]
            text_embedding (torch.Tensor): Text embeddings of shape [B, text_dim]
        Returns:
            torch.Tensor: Fused conditioning [B, C, T] (or [B, 2C, T] if 'concat')
        """
        B, C, T = c.shape
        assert C == self.c_dim, f"Expected c_dim={self.c_dim}, got {C}"

        # Project text embeddings from [B, text_dim] -> [B, c_dim]
        text_proj = self.text_mlp(text_embedding)  # [B, C]

        # Expand to match c's temporal dimension: [B, C, T]
        text_proj_expanded = text_proj.unsqueeze(-1).expand(-1, -1, T)

        # Fusion
        if self.fusion_type == 'add':
            fused = c + text_proj_expanded
        elif self.fusion_type == 'concat':
            fused = torch.cat([c, text_proj_expanded], dim=1)  # [B, 2C, T]
        elif self.fusion_type == 'gated_add':
            if self.pool_type == 'mean':
                pooled_c = c.mean(-1)
            elif self.pool_type == 'max':
                pooled_c, _ = c.max(-1)

            gate_input = torch.cat([pooled_c, text_proj], dim=-1)  # [B, 2C]
            gate = self.gate_layer(gate_input).unsqueeze(-1)  # [B, C, 1]
            fused = gate * text_proj_expanded + (1 - gate) * c
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        return fused
