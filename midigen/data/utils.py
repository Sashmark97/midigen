import torch
import random

from midigen.utils.constants import TOKEN_PAD, TORCH_LABEL_TYPE, TOKEN_END, SEQUENCE_START, CPU_DEVICE


def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=CPU_DEVICE)
    tgt = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=CPU_DEVICE)

    raw_len = len(raw_mid)
    full_seq = max_seq + 1  # Performing seq2seq

    if raw_len == 0:
        return x, tgt

    if raw_len < full_seq:
        if tgt.shape[0] == raw_len:
            x[:raw_len] = raw_mid
            tgt[:raw_len - 1] = raw_mid[1:]
            tgt[raw_len - 1] = TOKEN_END
        else:
            x[:raw_len] = raw_mid
            tgt[:raw_len - 1] = raw_mid[1:]
            tgt[raw_len] = TOKEN_END
    else:
        # Randomly selecting a range
        if random_seq:
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    return x, tgt
