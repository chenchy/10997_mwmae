import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import ml_collections
from ml_collections import ConfigDict
from . import models
import functools


def get_model_cls(config: ConfigDict):
    model_cls = getattr(models, config.model.arch)
    model_args = config.model.get("model_args", None)

    # print(model_args.to_dict())
    if model_args:
        model_cls = functools.partial(model_cls, **model_args.to_dict())
    print(model_cls)
    return model_cls


def create_model(model_cls, patch_embed_projection_module, 
                 half_precision, frontend_cls=None, invert_features=False, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    print("!!!!!!!!!!!!!!!!!!!!!!!!!! in create_model, model_dtype", model_dtype)
    if frontend_cls is None:
        print("!!! frontend is none..")
        return model_cls(patch_embed_projection_module=patch_embed_projection_module, 
                         dtype=model_dtype, **kwargs)
    else:
        mcls = functools.partial(model_cls, patch_embed_projection_module=patch_embed_projection_module, 
                        dtype=model_dtype, **kwargs)
        return models.LearnableFrontendWrapper(model_cls=mcls, frontend_cls=frontend_cls, dtype=model_dtype,
                                               invert_features=invert_features)


def get_patch_embed(config: ml_collections.ConfigDict):
    if "raw" in config.model.arch:
        patch_embed_args = config.model.get("patch_embed_args", None)
        # patch_embed_args = config.model.patch_embed_args
        if patch_embed_args:
            return models.patch_embed_projection_module(patch_embed_args.to_dict(),
                                                        sample_rate=config.audio_config.get("sample_rate", 16000))
        else:
            return None
    else:
        return None


def precomputed_feature_extract_fn(x, dtype=jnp.float32, mean=None, std=None):
    x = x[Ellipsis, jnp.newaxis]
    if mean is not None and std is not None:
        print("Normalizing!!!!!!!!!! x with shape:", x.shape)
        x = (x - mean) / std
    x = x.astype(dtype)
    return x
