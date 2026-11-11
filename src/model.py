"""Neural architectures for numerical tabular classification.

The hybrid classifier and tabular ResNet are custom extensions of the
FT-Transformer-inspired architecture.

References
----------
Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A.
"Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021.
https://arxiv.org/abs/2106.11959

PyTorch TransformerEncoderLayer documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

PyTorch TransformerEncoder documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

PyTorch BatchNorm1d documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

PyTorch Linear documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""

import torch
from torch import nn


class NumericalFeatureTokenizer(nn.Module):
    """Convert each numerical feature into an independently learned token."""

    def __init__(self, num_features: int = 14, token_dim: int = 192) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")

        self.num_features = num_features
        self.token_dim = token_dim
        self.weight = nn.Parameter(torch.empty(num_features, token_dim))
        self.bias = nn.Parameter(torch.empty(num_features, token_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the feature-specific affine parameters."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Tokenize a numerical tensor of shape ``[B, num_features]``."""
        if inputs.ndim != 2:
            raise ValueError(
                "Expected a 2D input tensor with shape [batch_size, num_features], "
                f"but received {inputs.ndim} dimensions"
            )
        if inputs.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but received {inputs.shape[1]}"
            )

        # [B, F] -> [B, F, 1], then apply one learned affine map per feature.
        return inputs.unsqueeze(-1) * self.weight + self.bias


class FTTransformerClassifier(nn.Module):
    """FT-Transformer-inspired classifier for numerical tabular data."""

    def __init__(
        self,
        num_features: int = 14,
        num_classes: int = 3,
        token_dim: int = 192,
        num_heads: int = 8,
        ffn_dim: int = 384,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.num_features = num_features
        self.tokenizer = NumericalFeatureTokenizer(num_features, token_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a batch of numerical features."""
        if inputs.ndim != 2:
            raise ValueError(
                "Expected a 2D input tensor with shape [batch_size, num_features], "
                f"but received {inputs.ndim} dimensions"
            )
        if inputs.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but received {inputs.shape[1]}"
            )

        # Numerical features: [B, F] -> feature tokens: [B, F, D].
        feature_tokens = self.tokenizer(inputs)
        # Expand the learned token across the batch: [1, 1, D] -> [B, 1, D].
        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        # Prepend CLS to form the encoder sequence: [B, F + 1, D].
        sequence = torch.cat((cls_tokens, feature_tokens), dim=1)
        encoded = self.encoder(sequence)
        # Classify only the final CLS representation: [B, D] -> [B, C].
        return self.head(encoded[:, 0])


class HybridFTTransformerClassifier(nn.Module):
    """Custom MLP extension of the existing FT-Transformer-inspired model."""

    def __init__(
        self,
        num_features: int = 14,
        num_classes: int = 3,
        token_dim: int = 192,
        num_heads: int = 8,
        ffn_dim: int = 384,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.num_features = num_features
        self.tokenizer = NumericalFeatureTokenizer(num_features, token_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_branch = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim + 128),
            nn.Linear(token_dim + 128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw logits from parallel Transformer and MLP branches."""
        if inputs.ndim != 2:
            raise ValueError(
                "Expected a 2D input tensor with shape [batch_size, num_features], "
                f"but received {inputs.ndim} dimensions"
            )
        if inputs.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but received {inputs.shape[1]}"
            )

        # Transformer branch: [B, F] -> [B, F, D].
        feature_tokens = self.tokenizer(inputs)
        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        sequence = torch.cat((cls_tokens, feature_tokens), dim=1)
        # Retain the final CLS representation: [B, F + 1, D] -> [B, D].
        cls_representation = self.encoder(sequence)[:, 0]

        # Direct normalized-feature MLP branch: [B, F] -> [B, 128].
        mlp_representation = self.mlp_branch(inputs)
        # Fuse both branches before classification: [B, D + 128].
        combined = torch.cat((cls_representation, mlp_representation), dim=1)
        return self.head(combined)


class TabularResidualBlock(nn.Module):
    """Apply a residual MLP transformation at a fixed feature width."""

    def __init__(
        self,
        hidden_dim: int = 256,
        block_hidden_dim: int = 512,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        norm_layer = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
        self.block = nn.Sequential(
            norm_layer,
            nn.Linear(hidden_dim, block_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the residual sum for a tensor of shape ``[B, D]``."""
        return inputs + self.block(inputs)


class TabularResNetClassifier(nn.Module):
    """Classify numerical tables with a custom residual MLP."""

    def __init__(
        self,
        num_features: int = 14,
        num_classes: int = 3,
        hidden_dim: int = 256,
        block_hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if block_hidden_dim <= 0:
            raise ValueError("block_hidden_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.num_features = num_features
        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.residual_blocks = nn.Sequential(
            *[
                TabularResidualBlock(hidden_dim, block_hidden_dim, dropout, use_layer_norm)
                for _ in range(num_blocks)
            ]
        )
        head_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
        self.head = nn.Sequential(
            head_norm,
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a 2D numerical feature tensor."""
        if inputs.ndim != 2:
            raise ValueError(
                "Expected a 2D input tensor with shape [batch_size, num_features], "
                f"but received {inputs.ndim} dimensions"
            )
        if inputs.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but received {inputs.shape[1]}"
            )

        # Project [B, F] to [B, D] and preserve that width through all blocks.
        representation = self.input_projection(inputs)
        representation = self.residual_blocks(representation)
        # Produce unnormalized class scores: [B, D] -> [B, C].
        return self.head(representation)
