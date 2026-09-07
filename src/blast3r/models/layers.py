# Copyright (C) 2026-present Naver Corporation. All rights reserved.

from functools import partial

import torch
import torch.nn as nn

from .blocks import Mlp, Attention, CrossAttention
from .layer_scale import LayerScale
from .pos_embed import cuRoPE100

class Network (nn.Module):
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.bias is not None: # is not None in case elementwise_affine=False
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class PositionAugmentor(nn.Module):
    """Rescales token positions into the range the model was trained on."""
    def __init__(self, posaug_factor=None):
        super().__init__()
        self.posaug_factor = posaug_factor
        self.auto_resize = None

    def forward(self, pos):
        if self.posaug_factor is not None:
            if self.auto_resize is not None:
                assert 30 < self.auto_resize < 50, 'seems like plausible range'
                pos = pos * (self.auto_resize-1) / pos.max()

            pos = pos / self.posaug_factor # avoids to be too much out of training range at test time, not sure it is actually necessary
        return pos

class ForegroundManaging_Patchifier(Network):
    """Replace the RGB value of background pixels with bg_bias."""
    def __init__(self, dim=None, manage_foreground=False):
        super().__init__()
        self.manage_foreground = manage_foreground
        self.bg_bias = -1.1 # special value out of RGB range

    def apply_mask(self, img, foreground_mask):
        if self.manage_foreground:
            has_fg, fg = foreground_mask
            has_fg = has_fg[:,0]
            background = ~fg[has_fg].unsqueeze(2).expand(-1,-1,3,-1,-1)
            img[has_fg][background] = self.bg_bias
        return img

class LinearPatchifier (ForegroundManaging_Patchifier):
    def __init__(self, patch_size, dim=256, ichan=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), manage_foreground=False):
        super().__init__(dim, manage_foreground)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(ichan, dim, self.patch_size, stride=self.patch_size, bias=True)
        self.proj_ln = norm_layer(dim)
        self.dim = dim

    def forward(self, img, foreground_mask=(None, None)):
        B,ntokens,THREE,ph,pw = img.size()
        x = self.proj(self.apply_mask(img, foreground_mask).view(B*ntokens,THREE,ph,pw)).view(B,ntokens,self.dim)
        x = self.proj_ln(x)
        return x

class LinearHead (nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim or in_dim)

    def forward(self, x):
        return self.head(x)


class MlpHead (nn.Module):
    def __init__(self, in_dim, out_dim=None, mlp_mul=4):
        super().__init__()
        self.head = Mlp(in_dim, mlp_mul*in_dim, out_dim or in_dim)

    def forward(self, x):
        return self.head(x)


class MyBlock (nn.Module):
    def __init__(self, dim, num_heads=None,
                 with_SA=True, with_CA=True, with_CA_rope=True, with_MLP=True,
                 mlp_ratio=4., pos_embed=None, qkv_bias=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, qkln=False, split_qkv=False, init_values=None):
        super().__init__()
        if with_CA:
            self.ls_ca = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.norm_y = norm_layer(dim)
            self.norm_ca = norm_layer(dim)
            AttnBlock = default(with_CA, CrossAttention)
            self.cross_attn = AttnBlock(dim, pos_embed=pos_embed if with_CA_rope else None, num_heads=num_heads, qkv_bias=qkv_bias, qkln=qkln)
        else:
            self.cross_attn = False

        if with_SA:
            self.ls_sa = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.norm_sa = norm_layer(dim)
            AttnBlock = default(with_SA, Attention)
            self.attn = AttnBlock(dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias, qkln=qkln, split_qkv=split_qkv)
        else:
            self.attn = False

        if with_MLP:
            self.ls_mlp = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.norm_mlp = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            MlpBlock = default(with_MLP, Mlp)
            self.mlp = MlpBlock(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        else:
            self.mlp = False

    def forward(self, x, xpos=None, y=None, ypos=None, sa_attn_mask=None, ca_attn_mask=None, mlp_mask=None):
        if x.numel() == 0:
            return x # to avoid crashing in the attention

        if self.attn: # self-attention
            x = x + self.ls_sa(self.attn(self.norm_sa(x), xpos, mask_for_attention=sa_attn_mask))

        if self.cross_attn: # cross-attention
            y = self.norm_y(y)
            x = x + self.ls_ca(self.cross_attn(self.norm_ca(x), y, y, xpos, ypos, mask_for_attention=ca_attn_mask))

        if self.mlp:
            if mlp_mask is not None:
                raise NotImplementedError('masked MLP is not supported by the released models')
            x = x + self.ls_mlp(self.mlp(self.norm_mlp(x)))
        return x


def default(block, default_block):
    return default_block if block is True else block


class ViT (Network):
    """Basic ViT."""
    def __init__(self, patch_size=None, output_dim=None, dim=256, depth=4, Patchifier=LinearPatchifier, posaug_factor=None,
                 with_SA=True, with_CA=False, with_CA_rope=False, with_MLP=True, Head=LinearHead,
                 head_size=64, qkln=False, PosEmbed = cuRoPE100, null_tokens = 0,
                 mlp_ratio=4, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 manage_foreground=False, split_qkv=False, init_values=None):
        super().__init__()
        self.num_heads = dim//head_size
        assert self.num_heads * head_size == dim

        # project to token dimension
        self.patch_size = patch_size
        self.patchify = Patchifier(patch_size, dim, manage_foreground=manage_foreground) if None not in (patch_size, Patchifier) else None
        self.null_tokens = nn.Parameter(torch.randn((null_tokens, dim)))
        self.pos_augmentor = PositionAugmentor(posaug_factor)

        # vit
        pos_embed = PosEmbed(dim, self.num_heads, num_cls=0)
        self.blocks = nn.ModuleList([
                MyBlock(dim, self.num_heads, with_SA=with_SA, with_CA=with_CA, with_CA_rope=with_CA_rope,
                        with_MLP=with_MLP, mlp_ratio=mlp_ratio, act_layer=act_layer,
                        norm_layer=norm_layer, pos_embed=pos_embed, qkv_bias=True, qkln=qkln, split_qkv=split_qkv, init_values=init_values)
                for i in range(depth)])
        self.norm = norm_layer(dim)

        # projection head
        self.headless = not(output_dim) or (Head is None)
        if not self.headless:
            self.head = Head(dim, output_dim)

        # initialize weights
        self.apply(self._init_weights)

    def _make_null_tokens(self, input_shape, *B):
        B1 = tuple(1 for _ in B)
        n_cls = len(self.null_tokens)
        null_tokens = self.null_tokens.view(*B1,n_cls,self.null_tokens.shape[-1]).expand(*B,-1,-1)

        # prepare null CLS positions:
        # first 4 tokens are in the corners, then randomly everywhere
        H, W = input_shape.split(1, dim=-1)
        null_pos = -torch.ones( (*B,n_cls,2), dtype=torch.float32, device=self.device)
        null_pos[:, [1,3],1] = W.float()
        null_pos[:, [2,3],0] = H.float()
        if n_cls > 4:
            null_pos[:, 4:, 0] = H * torch.rand((*B,n_cls-4), device=self.device)
            null_pos[:, 4:, 1] = W * torch.rand((*B,n_cls-4), device=self.device)

        return null_tokens.clone(), null_pos

    def forward(self, x, pos, true_shape=None, ret_x=False, sa_attn_mask=None, ca_attn_mask=None, return_patchified=False, foreground_mask=(None, None), **ca_tokens):
        # patchify
        input_shape = None
        if self.patchify is not None:
            x = self.patchify(x, foreground_mask)
            pos = self.pos_augmentor(pos)
            input_shape = true_shape // self.patch_size # (h, w)

            if len(self.null_tokens):
                null_tokens, null_pos = self._make_null_tokens(input_shape, *x.shape[:-2])
                x = torch.cat((null_tokens, x), dim=-2)
                pos = torch.cat((null_pos, pos), dim=-2)

        if return_patchified:
            patchx = x.clone()

        # encoder
        all_x = [x]
        for blk in self.blocks:
            x = blk(x, pos, sa_attn_mask=sa_attn_mask, ca_attn_mask=ca_attn_mask, **ca_tokens)
            all_x.append(x)

        dec = self.post_dec(all_x, pos, input_shape, ret_x=ret_x)
        return dec if not return_patchified else (dec, patchx)

    def post_dec(self, all_x, pos, input_shape, ret_x=False):
        y = self.norm(all_x[-1])

        # return intermediate layer's tokens?
        x = all_x if ret_x == 'all' \
            else all_x[-1] if ret_x is True \
            else [all_x[i] for i in ret_x] if ret_x \
            else y # default is standard output

        if self.headless: # raw output if no head
            return (x, pos, input_shape) if input_shape is not None else (x, y)
        else:
            res = self.head(y) # (B, N, D)
            return (x, res) if ret_x else res
