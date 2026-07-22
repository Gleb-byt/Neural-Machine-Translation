"""
configuration of that project
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path

if not torch.cuda.is_available():
    raise ValueError("video card is not avialable")

print(torch.cuda.get_device_name())


@dataclass
class Config:
    devise: str = "cuda"

    """
    Data parameters
    """
    dataset_name: str = "Helsinki-NLP/opus-100"
    lang_pair: str = "en-ru"
    src_lang: str = "en"
    tgt_lang: str = "ru"
    max_seq_len: int = 128

    """
    tokenizer parameters
    """
    src_vocab_size: int = 16000
    tgt_vocab_size: int = 16000
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    """
    Model parameters
    """
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    """
    Training parameters
    """
    batch_size: int = 128
    num_epochs: int = 30
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    use_amp: bool = True

    """
    Paths
    """
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def data_raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def tokenizer_dir(self) -> Path:
        return self.base_dir / "tokenizer"

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def src_tokenizer_prefix(self) -> str:
        return str(self.tokenizer_dir / "src")

    @property
    def tgt_tokenizer_prefix(self) -> str:
        return str(self.tokenizer_dir / "tgt")

    def create_dirs(self) -> None:
        for d in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.tokenizer_dir,
            self.checkpoint_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
