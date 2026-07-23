from torch.nn.utils.rnn import pad_sequence


class Collator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        src_list = [item[0] for item in batch]
        tgt_list = [item[1] for item in batch]

        src_tensor = pad_sequence(src_list, batch_first=True, padding_value=self.pad_id)
        tgt_tensor = pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_id)

        return src_tensor, tgt_tensor
