import torch
from torch import Tensor

def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, dim_feedforward, enable_rpr=False, er_len=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim_feedforward = dim_feedforward
        self.enable_rpr = enable_rpr
        self.er_len = er_len
        for k,v in kwargs.items():
            setattr(self, k, v)
