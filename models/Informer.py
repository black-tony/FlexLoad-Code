import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.masking import TriangularCausalMask, ProbMask
if True:
    from Informer_assets.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
    from Informer_assets.decoder import Decoder, DecoderLayer
    from Informer_assets.attn import FullAttention, ProbAttention, AttentionLayer
    from Informer_assets.embed import DataEmbedding

    class Informer(nn.Module):
        def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                    factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                    dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                    output_attention = False, distil=True, mix=True,
                    device=torch.device('cuda:0')):
            super(Informer, self).__init__()
            self.pred_len = out_len
            self.attn = attn
            self.output_attention = output_attention

            # Encoding
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
            # Attention
            Attn = ProbAttention if attn=='prob' else FullAttention
            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(e_layers-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            )
            # Decoder
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                    d_model, n_heads, mix=mix),
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
            # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
            # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
            self.projection = nn.Linear(d_model, c_out, bias=True)
            
        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_out = self.projection(dec_out)
            
            # dec_out = self.end_conv1(dec_out)
            # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
            if self.output_attention:
                return dec_out[:,-self.pred_len:,:], attns
            else:
                return dec_out[:,-self.pred_len:,:] # [B, L, D]


    class InformerStack(nn.Module):
        def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                    factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                    dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                    output_attention = False, distil=True, mix=True,
                    device=torch.device('cuda:0')):
            super(InformerStack, self).__init__()
            self.pred_len = out_len
            self.attn = attn
            self.output_attention = output_attention

            # Encoding
            self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
            # Attention
            Attn = ProbAttention if attn=='prob' else FullAttention
            # Encoder

            inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                        d_model, n_heads, mix=False),
                            d_model,
                            d_ff,
                            dropout=dropout,
                            activation=activation
                        ) for l in range(el)
                    ],
                    [
                        ConvLayer(
                            d_model
                        ) for l in range(el-1)
                    ] if distil else None,
                    norm_layer=torch.nn.LayerNorm(d_model)
                ) for el in e_layers]
            self.encoder = EncoderStack(encoders, inp_lens)
            # Decoder
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                    d_model, n_heads, mix=mix),
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
            # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
            # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
            self.projection = nn.Linear(d_model, c_out, bias=True)
            
        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_out = self.projection(dec_out)
            
            # dec_out = self.end_conv1(dec_out)
            # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
            if self.output_attention:
                return dec_out[:,-self.pred_len:,:], attns
            else:
                return dec_out[:,-self.pred_len:,:] # [B, L, D]

if False:
    class AttentionLayer(nn.Module):
        def __init__(self, embed_size, heads, dropout=0.1):
            super(AttentionLayer, self).__init__()
            self.attn = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        
        def forward(self, query, key, value):
            output, _ = self.attn(query, key, value)
            return output

    class InformerEncoderLayer(nn.Module):
        def __init__(self, embed_size, heads, ff_hid_dim, dropout=0.1):
            super(InformerEncoderLayer, self).__init__()
            self.attn = AttentionLayer(embed_size, heads, dropout)
            self.ffn = nn.Sequential(
                nn.Linear(embed_size, ff_hid_dim),
                nn.ReLU(),
                nn.Linear(ff_hid_dim, embed_size)
            )
            self.layernorm1 = nn.LayerNorm(embed_size)
            self.layernorm2 = nn.LayerNorm(embed_size)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            attn_out = self.attn(x, x, x)
            x = self.layernorm1(x + self.dropout(attn_out))
            
            ffn_out = self.ffn(x)
            x = self.layernorm2(x + self.dropout(ffn_out))
            
            return x

    class Informer(nn.Module):
        def __init__(self, input_size, embed_size, hidden_size, num_layers, num_heads, ff_hid_dim=2048, dropout=0.1):
            super(Informer, self).__init__()
            
            self.input_size = input_size
            self.embed_size = embed_size
            self.fc_input = nn.Linear(input_size, embed_size)

            self.encoder_layers = nn.ModuleList([
                InformerEncoderLayer(embed_size, num_heads, ff_hid_dim, dropout) for _ in range(num_layers)
            ])
            
            self.fc_output = nn.Linear(embed_size, input_size)  # Output size matches input_size

        def forward(self, x):
            x = self.fc_input(x)

            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
            
            out = x[:, -1, :]
            
            out = self.fc_output(out)
            
            return out
