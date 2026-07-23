import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from embeddings import TransformerEmbedding
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.src_emb = TransformerEmbedding(src_vocab_size, d_model)
        self.tgt_emb = TransformerEmbedding(tgt_vocab_size, d_model)

        self.pe = PositionalEncoding(d_model)

        self.encoder = Encoder(num_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, n_heads, d_ff, dropout)

        self.output_head = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.src_emb(src)
        src_emb = self.pe(src_emb)
        src_emb = self.dropout(src_emb)

        enc_out = self.encoder(src_emb, src_mask)

        tgt_emb = self.tgt_emb(tgt)
        tgt_emb = self.pe(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        dec_out = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)

        return self.output_head(dec_out)
