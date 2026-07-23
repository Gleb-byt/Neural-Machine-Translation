from typing import List
import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:

        ids = self.sp.encode(text, out_type=int)

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int]) -> str:

        special = {self.pad_id, self.bos_id, self.eos_id}
        filter_ids = [i for i in ids if i not in special]

        return self.sp.decode(filter_ids)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:

        return [self.encode(text, add_special_tokens) for text in texts]

    def id_to_piece(self, one_id: int) -> str:
        return self.sp.id_to_piece(one_id)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab size = {self.vocab_size})"


if __name__ == "__main__":
    import pathlib

    path = pathlib.Path(__file__).parent / "src.model"
    T = Tokenizer(str(path))
    sentence = "Hello WORLD"
    t_sentence_int = T.encode(sentence)
    t_sentence_str = T.tokenize(sentence)

    sentence_decode = T.decode(t_sentence_int)

    print(t_sentence_int, t_sentence_str)
    print(sentence_decode)

    another_sentence = "переизобрести"

    new_path = pathlib.Path(__file__).parent / "tgt.model"
    T_ru = Tokenizer(str(path))

    t_ru_s = T.encode(another_sentence)
    t_ru_s_str = T.tokenize(another_sentence)

    print(t_ru_s, t_ru_s_str)
