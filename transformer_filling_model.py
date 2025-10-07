import torch
import torch.nn as nn

class TransformerFillingModel(nn.Module):
    def __init__(self, num_feats, embed_dim, num_layers=8, dropout=0.1, **kwargs):
        super(TransformerFillingModel, self).__init__()
        
        self.num_feats = num_feats
        self.embed_dim = embed_dim

        # Embedding list: one embedding per feature (vocab size = 9 for tokens 0–8)
        self.emblist = nn.ModuleList([
            nn.Embedding(32, embed_dim) for _ in range(num_feats)
        ])

        # Simple learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_feats, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True  # input: (B, seq_len, embed_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Linear layers for each feature: shape (embed_dim → 8 possible output values)
        self.output_heads = nn.ModuleList([
            nn.Linear(embed_dim, 16) for _ in range(num_feats)
        ])

    def forward(self, x):
        """
        x: (B, num_feats) tensor with integer tokens 0–8
        Returns: (B, num_feats, 8) logits
        """
        batch_size = x.size(0)

        # Apply feature-wise embeddings
        embedded_feats = []

        for i in range(self.num_feats):
            emb = self.emblist[i](x[:, i])  # (B, embed_dim)
            embedded_feats.append(emb)
        embedded_feats = torch.stack(embedded_feats, dim=1)  # (B, num_feats, embed_dim)

        # Add positional embeddings
        embedded_feats = embedded_feats + self.pos_embedding

        # Transformer encoder
        encoded = self.transformer_encoder(embedded_feats)  # (B, num_feats, embed_dim)

        # Apply feature-specific output heads
        outputs = []
        for i in range(self.num_feats):
            logits = self.output_heads[i](encoded[:, i, :])  # (B, 8)
            outputs.append(logits)
        outputs = torch.stack(outputs, dim=1)  # (B, num_feats, 8)

        return outputs

