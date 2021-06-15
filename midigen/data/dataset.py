import torch
import joblib
from tqdm import tqdm
from torch.utils.data import Dataset

from midigen.utils.constants import TORCH_LABEL_TYPE, CPU_DEVICE
from midigen.data.utils import process_midi


class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, file_list, max_seq=2048, random_seq=True, num_files=-1, type='training'):
        self.max_seq = max_seq
        self.random_seq = random_seq
        self.data_files = []
        for file in tqdm(file_list[:num_files], desc=f'Loading {type} dataset: '):
            seqs = joblib.load(file)
            for seq in seqs:
                if len(seq) == 0:
                    continue
                self.data_files.append(seq)
        # CPU device to enable parallel operations
        self.device = CPU_DEVICE

    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.
        Returns the input and the target.
        ----------
        """

        raw_mid = torch.tensor(self.data_files[idx], dtype=TORCH_LABEL_TYPE, device=self.device)
        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return x, tgt
