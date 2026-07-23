import torch
from torch.utils.data import Dataset
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from tokenizer.tokenizer import Tokenizer


class Translationdataset(Dataset):
    def __init__(
        self, src_path, tgt_path, src_tok: Tokenizer, tgt_tok: Tokenizer, max_len=128
    ):
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_data = f.readlines()

        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_data = f.readlines()

        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx) -> tuple[torch.types.Tensor, torch.types.Tensor]:

        src_ids = self.src_tok.encode(self.src_data[idx].strip())[: self.max_len]
        tgt_ids = self.tgt_tok.encode(self.tgt_data[idx].strip())[: self.max_len]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)
