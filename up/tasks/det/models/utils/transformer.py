import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 static=False, seq_l=196):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.static = static
        if self.static:
            self.static_a = nn.Parameter(torch.Tensor(1, num_heads, seq_l, seq_l))
            trunc_normal_(self.static_a)
        self.custom_flops = 2 * seq_l * seq_l * dim

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if mask_type:
            mask = torch.ones_like(qkv)
            mask[:, :, head] = 0
            if mask_type == 'layer':
                mask = 1 - mask
            qkv = qkv * mask
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.static:
            attn = attn + self.static_a
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, activation='relu', dropout=0., norm_style='pre_norm'):
        super().__init__()
        assert norm_style in ['pre_norm', 'post_norm']
        self.norm_style = norm_style

        self.multihead_att = Attention(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        if self.norm_style == 'pre_norm':
            return self.pre_norm_forward(x)
        elif self.norm_style == 'post_norm':
            return self.post_norm_forward(x)
        else:
            raise NotImplementedError

    def pre_norm_forward(self, x):
        x_ = self.norm1(x)
        x = x + self.dropout1(self.multihead_att(x_))

        x_ = self.norm2(x)
        x_ = self.linear2(self.dropout(self.act(self.linear1(x_))))
        x = x + self.dropout2(x_)
        return x

    def post_norm_forward(self, x):
        x_ = self.multihead_att(x)
        x = x + self.dropout1(x_)
        x = self.norm1(x)

        x_ = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = x + self.dropout2(x_)
        x = self.norm2(x)
        return x