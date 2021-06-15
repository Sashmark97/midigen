import torch.nn as nn

from midigen.models.musictransformer.rpr import MultiheadAttentionRPR
from midigen.models.gpt2.attention import CausalSelfAttention

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.enable_rpr = config.enable_rpr
        if config.enable_rpr:
            self.attn = MultiheadAttentionRPR(config.n_embd, config.n_head, config.attn_pdrop, er_len=config.er_len)
        else:
            self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask=None):
        if self.enable_rpr:
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        else:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
