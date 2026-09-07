# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# The Attention and Block classes are a modification of timm
# (https://github.com/huggingface/pytorch-image-models),
# timm/models/vision_transformer.py, licensed under the Apache License 2.0,
# with a decoder that takes an additional input and performs self-attention,
# cross-attention and mlp.
# SwiGLU is adapted from PaLM-pytorch (https://github.com/lucidrains/PaLM-pytorch),
# palm_pytorch/palm_pytorch.py, licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers.helpers import to_2tuple
except ModuleNotFoundError:
    from timm.models.layers import to_2tuple

try:
    import xformers.ops
    has_xformers = True
except Exception as e:
    has_xformers = False

_use_memory_efficient_attention = False
def toggle_memory_efficient_attention(enabled: bool = True):
    global _use_memory_efficient_attention
    _use_memory_efficient_attention = enabled

def is_memory_efficient_attention_enabled():
    return _use_memory_efficient_attention


class SwiGLU(nn.Module): # https://github.com/lucidrains/PaLM-pytorch/blob/7164d13d5a831647edb5838544017f387130f987/palm_pytorch/palm_pytorch.py#L61
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        if act_layer is SwiGLU: # to keep parameters similar to standard activation, we go to 2/3 (*2), see EVA-02
            hidden_features_out = (int(hidden_features * 2 / 3) + 7) // 8 * 8
            hidden_features_in = 2 * hidden_features_out
        else:
            hidden_features_out = hidden_features
            hidden_features_in = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features_in, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features_out, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CoreAttention (nn.Module):
    def __init__(self, pos_embed=None, attn_drop=0.):
        super().__init__()
        self.pos_embed = pos_embed
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_val = attn_drop

    def attention(self, q, k, v, qpos=None, kpos=None, mask_for_attention=None):
        assert q.ndim == k.ndim == v.ndim == 4
        B, H, Nq, C = q.shape
        Nk = k.shape[-2]
        assert k.shape == v.shape == (B, H, Nk, C)

        # first, apply RoPE inline
        q_ini = q
        k_ini = k
        if self.pos_embed is not None:
            # will copy if self.num_cls > 0
            q, _ = self.pos_embed.apply_after_qkv(q, qpos)
            k, _ = self.pos_embed.apply_after_qkv(k, kpos)

        num_cls = 0 if self.pos_embed is None else self.pos_embed.num_cls
        optimized = (is_memory_efficient_attention_enabled() and
                     num_cls == 0 )

        attn_bias = None
        if mask_for_attention is not None:
            attn_bias = torch.where(mask_for_attention[:,None], 0, -float('inf'))
            attn_bias = attn_bias.expand(-1, H, -1, -1)

        if optimized:
            assert has_xformers
            assert q.dtype == v.dtype
            assert k.dtype == v.dtype
            q, k, v = map(lambda val: val.reshape(B*H, -1, C), (q, k, v))
            if attn_bias is not None:
                attn_bias = attn_bias.reshape([q.shape[0],q.shape[1],k.shape[1]])
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.attn_drop_val)
            x = x.reshape(B, H, Nq, C)

        else:
            if num_cls > 0:
                """ Computing the attention between
                            | K_cls |    K_pos   |
                    |-------|-------|------------|
                    | Q_cls |       | sim_qcls_k |
                    |-------| sim_q |------------|
                    | Q_pos |       |  sim_rope  |
                    |-------|-------|------------|
                """
                # cls tokens are first
                sim = q[:, :, num_cls:] @ k[:, :, num_cls:].transpose(-2, -1)

                sim_qcls_k = q_ini[:, :, :num_cls] @ k_ini[:, :, num_cls:].transpose(-2, -1)
                sim_q = q_ini @ k_ini[:, :, :num_cls].transpose(-2, -1)
                sim = torch.concatenate([sim_qcls_k, sim], dim=-2)
                sim = torch.concatenate([sim_q, sim], dim=-1)
            else:
                sim = q @ k.transpose(-2, -1)

            sim *= self.scale
            if attn_bias is not None:
                sim += attn_bias

            attn = sim.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v) # (B, H, Nq, C)

        return x.transpose(1, 2).reshape(B, Nq, C*H)


class Attention (CoreAttention):
    """Attention with a position embedding acting inside the attention.

    Otherwise identical to timm.models.vision_transformer.Attention.
    """
    def __init__(self, dim, pos_embed=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qkln=False, split_qkv=False):
        super().__init__(pos_embed=pos_embed, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if split_qkv:
            self.projq = nn.Linear(dim, dim, bias=qkv_bias)
            self.projk = nn.Linear(dim, dim, bias=qkv_bias)
            self.projv = nn.Linear(dim, dim, bias=qkv_bias)
            self.qkv = lambda x: torch.cat([self.projq(x), self.projk(x), self.projv(x)], dim=-1)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkln = qkln
        if self.qkln:
            self.qln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)
            self.kln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)

    def forward(self, x, xpos, mask_for_attention=None):
        B, N, C = x.shape

        if self.qkln: # a bit ugly to have contiguous tensor for curope
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = [qkv.select(2, i) for i in range(3)]
            q = self.qln(q).transpose(1,2)
            k = self.kln(k).transpose(1,2)
            v = v.transpose(1,2)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
            q, k, v = [qkv.select(2, i) for i in range(3)]

        x = self.attention(q, k, v, xpos, xpos, mask_for_attention)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention (CoreAttention):

    def __init__(self, dim, pos_embed=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qkln=False,
                       kv_dim=None):
        super().__init__(pos_embed=pos_embed, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if kv_dim is None: kv_dim = dim

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkln = qkln
        if self.qkln:
            self.qln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)
            self.kln = nn.LayerNorm(head_dim, eps=1e-06, elementwise_affine=False)

    def forward(self, query, key, value, qpos, kpos, mask_for_attention=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        if self.qkln: # a bit ugly to have contiguous tensor for curope
            q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads)
            k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads)
            v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads)
            q = self.qln(q)
            k = self.kln(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        else:
            q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)

        x = self.attention(q, k, v, qpos, kpos, mask_for_attention=mask_for_attention)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
