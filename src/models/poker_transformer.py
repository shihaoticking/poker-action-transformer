import torch
import torch.nn as nn


class PokerTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        num_classes: int = 3,
    ):
        """
        Args:
            vocab_sizes (dict): Dictionary with vocab sizes for 'round', 'card', and 'action'.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Feedforward layer size in transformer block.
            dropout (float): Dropout rate.
            max_seq_len (int): Max sequence length.
            num_classes (int): Output classes (e.g., 3 for C, R, F).
        """
        super().__init__()
        self.d_model = d_model

        self.round_embed = nn.Embedding(vocab_sizes["round_vocab_size"], d_model)
        self.card_embed = nn.Embedding(vocab_sizes["card_vocab_size"], d_model)
        self.action_embed = nn.Embedding(vocab_sizes["action_vocab_size"], d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len)
            types: (batch_size, seq_len), 0: round, 1: card, 2: action
        Returns:
            Tensor: Output logits of shape (batch_size, num_classes)
        """

        x = torch.zeros(
            tokens.shape[0], tokens.shape[1], self.d_model, device=tokens.device
        )

        # Embed tokens by type
        for type_id, emb_layer in enumerate(
            [self.round_embed, self.card_embed, self.action_embed]
        ):
            mask = types == type_id
            if mask.any():
                x[mask] = emb_layer(tokens[mask])

        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)

        # Use final token representation
        x = x[:, -1, :]  # (batch_size, d_model)

        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits
