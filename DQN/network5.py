import math

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
# from attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.CBAM import *

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.Tanh(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerBlockwai(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlockwai, self).__init__()
        self.attentionwai = CBAMBlock(channel=embed_size,reduction=heads,kernel_size=1)#SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.Tanh(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        value1=value.reshape(value.shape[0],value.shape[2],int(math.sqrt(value.shape[1])),int(math.sqrt(value.shape[1])))
        attentionwai = self.attentionwai(value1)
        attentionwai=attentionwai.reshape(value.shape[0],value.shape[1],value.shape[2])
        x = self.dropout(self.norm1(attentionwai + value))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.init_x = torch.nn.Linear(src_vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlockwai(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        out = self.init_x(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # self.layers = nn.ModuleList(
        #     [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
        #      for _ in range(num_layers)]
        # )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)]
        )
        self.fc_out1 = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_x = torch.nn.Linear(9, 128)
        self.fc_out2 = nn.Linear(18, trg_vocab_size)

    def forward(self, x, enc_out,enc_out2, src_mask, trg_mask):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        # x=self.init_x(x)
        for layer in self.layers:
            x = layer(enc_out, enc_out2,x,src_mask)
        out = self.fc_out1(x)
        out =out.reshape(out.shape[0], 1, out.shape[1] * out.shape[2])
        # out = self.fc_out2(out)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=9, kernel_size=(65, 1), stride=1)
        self.lin = torch.nn.Linear(64, 1)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src,src2, trg):
        src_mask = None
        trg_mask = None
        enc_src = self.encoder(src, src_mask)
        enc_src2 = self.encoder(src2, src_mask)
        enc_src3 = self.encoder(trg, src_mask)
        # out = self.decoder(trg, enc_src,enc_src2, src_mask, trg_mask)
        # x0 = enc_src.unsqueeze(-1)
        # x1 = enc_src2.unsqueeze(-1)
        # x2 = enc_src3.unsqueeze(-1)
        # out= torch.cat([enc_src,enc_src2, enc_src3], 1)
        # out = out.unsqueeze(-1)
        # out = F.tanh(self.conv2(out))
        # out=out.squeeze(-1)
        # out=self.lin(out)
        out = self.decoder(enc_src3, enc_src, enc_src2, src_mask, trg_mask)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.tensor([[[1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.2265, 0.2265,0.2265],[0.5000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000,1.0000]]]).to(device)
    x1=torch.tensor([[[1.0000, 0.5000, 0.2500, 0.2500, 0.7500, 0.0000, 0.7500, 0.5000,0.0000]]]).to(device)
    trg = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0.]]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 9
    trg_vocab_size = 9
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x,x1, trg)
    out = out.view(-1, trg_vocab_size)
    # x = x.view(1, 32)
    # return self.fc(x)
    mask = torch.tensor([[1,1,1,1,1,1,1,1,1]], dtype=torch.float).to(device)
    # mask=x+mask.log()
    probs = torch.softmax(out + mask.log(), dim=1) # 概率P
    print(out)
    print(probs)
