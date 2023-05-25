# Add Attention, LayerScale, Block from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Sequence, Union, Callable, Optional
from .utils import constant_init
from .patch_embed import PatchEmbed
from .layers import Attention, LayerScale, Block, BNWrapper
from einops import rearrange

dense_kernel_init = nn.initializers.xavier_uniform()


class VisionTransformer(nn.Module):
    img_size: Union[Sequence, int] = (208, 80)
    patch_size: Union[Sequence, int] = (16, 16)
    in_chans: int = 1
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    norm_layer: Optional[Callable] = nn.LayerNorm
    global_pool: bool = False
    lin_probe: bool = False
    patch_embed_projection_module: Callable = None
    dtype: Any = jnp.float32

    def setup(self):
        self.patch_embed = PatchEmbed(img_size=self.img_size,
                                      patch_dim=self.patch_size, 
                                      embed_dim=self.embed_dim,
                                      dtype=self.dtype)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02, dtype=self.dtype),
                                    [1, 1, self.embed_dim])

        self.pos_embed = self.param("pos_embed", nn.initializers.normal(0.02, dtype=self.dtype),
                               [1, self.num_patches+1, self.embed_dim])
        # TODO: make a drop out wrapper like BNWrapper above to allow initializing
        #
        # if self.drop_rate > 0.:
        #     self.pos_drop = partial(nn.Dropout, rate=self.drop_rate, name="pos_drop")
        dpr = [x for x in np.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = [
            Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                  qkv_bias=self.qkv_bias, drop=self.drop_rate,
                  attn_drop=self.attn_drop_rate, drop_path=dpr[i],
                  norm_layer=self.norm_layer, dtype=self.dtype,
                  name="encoder_block_{:02d}".format(i))
            for i in range(self.depth)
        ]
        self.encoder_norm = self.norm_layer(dtype=self.dtype, param_dtype=self.dtype, name="encoder_norm")
        if self.global_pool:
            self.fc_norm = self.norm_layer(dtype=self.dtype, param_dtype=self.dtype, name="fc_norm")
        # if self.lin_probe:
            # BatchNorm without affine was used for lin-probe in MAE
            # self.head_norm = BNWrapper(use_bias=False, use_scale=False, name='head_norm')
            # self.head_norm = self.norm_layer(name="head_norm")
            # self.head_norm = partial(nn.BatchNorm, use_bias=False,
            #                          use_scale=True, )

        self.head = nn.Dense(self.num_classes, dtype=self.dtype, param_dtype=self.dtype)


    def forward_features(self, x, train: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # if self.drop_rate > 0.:
        #     x = self.pos_drop(deterministic=not train)(x)
        for blk in self.blocks:
            x = blk(x, train=train)
        x = self.encoder_norm(x)

        outcome = x[:, 1:, :]
        grid_size = self.patch_embed.grid_size
        t, f = grid_size
        outcome = rearrange(outcome, 'b (t f) d -> b t (f d)', f=f, d=self.embed_dim)
        return outcome

    def __call__(self, x, train: bool = True):
        x = x.astype(self.dtype)
        x = x[Ellipsis, jnp.newaxis] if x.ndim == 2 else x
        x = self.forward_features(x, train)
        if self.global_pool:
            x = self.fc_norm(x)
        # if self.lin_probe:
        #     x = self.head_norm(x)
        x = self.head(x)
        x = x.astype(self.dtype)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_base_custom(**kwargs):
    model = VisionTransformer(
        patch_size=400, embed_dim=512, depth=6, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_base_raw_400(**kwargs):
    model = VisionTransformer(
        patch_size=400, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_base_raw_800(**kwargs):
    model = VisionTransformer(
        patch_size=800, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_msm_base_200_patch16x4(**kwargs):
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=(4, 16),
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_base_200_patch16x2(**kwargs):
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=(2, 16),
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_base_80_patch16x4(**kwargs):
    model = VisionTransformer(
        img_size=(80, 80),
        patch_size=(4, 16),
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_base_204_patch16x4(**kwargs):
    model = VisionTransformer(
        img_size=(204, 80),
        patch_size=(4, 16),
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_base_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_tiny_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_small_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_medium_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_large_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model


def vit_msm_huge_200(**kwargs):
    patch_size = kwargs.pop("patch_size", (4, 16))
    # decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    # decoder_depth = kwargs.pop("decoder_depth", 4)
    # decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    model = VisionTransformer(
        img_size=(200, 80),
        patch_size=patch_size,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    return model
