import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm
from config import Config


def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
) -> None:
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=character_coverage,
        normalization_rule_name="nmt_nfkc_cf",
        max_sentence_length=4192,
        input_sentence_size=0,
    )

    print(f"Finish of studying learning process of tokenizer {model_prefix}.model")


def main():
    config = Config()
    config.create_dirs()
    train_en = config.data_raw_dir / f"train.{config.src_lang}"
    train_ru = config.data_raw_dir / f"train.{config.tgt_lang}"

    if not train_en.exists() or not train_ru.exists():
        print("No necessary data exists")
        sys.exit(1)

    print("en tokenizer")
    train_sentencepiece(
        input_file=str(train_en),
        model_prefix=config.src_tokenizer_prefix,
        vocab_size=config.src_vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )

    print("ru tokenizer")

    train_sentencepiece(
        input_file=str(train_ru),
        model_prefix=config.tgt_tokenizer_prefix,
        vocab_size=config.tgt_vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
    )

    print("tokenizers learning complete")


if __name__ == "__main__":
    main()
