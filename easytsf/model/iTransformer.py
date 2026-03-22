import torch
import torch.nn as nn

from easytsf.layer.transformer import Encoder, EncoderLayer, FullAttention, AttentionLayer, iTransformer_Embedder


class iTransformer(nn.Module):
    def __init__(self, hist_len, pred_len, output_attention, d_model, dropout, factor, n_heads,
                 d_ff, activation, e_layers):
        super().__init__()
        self.seq_len = hist_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = iTransformer_Embedder(self.seq_len, d_model, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, var_x, marker_x):
        # import pdb; pdb.set_trace()
        # x_enc = var_x[..., 0]
        # x_mark_enc = marker_x[..., 0,]
        dec_out = self.forecast(var_x, marker_x)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
