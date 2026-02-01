import torch
import torch.nn as nn

class ResTransBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, channels)
        )

        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Flatten spatial dims
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        # Self-attention
        attn_out, _ = self.attn(
            self.norm1(x_flat),
            self.norm1(x_flat),
            self.norm1(x_flat)
        )

        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        # Restore spatial dims
        x_out = x_flat.permute(0, 2, 1).view(B, C, H, W)

        return x_out