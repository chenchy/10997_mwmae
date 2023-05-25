import code
import jax
import numpy as np
import optax
import flax.linen as nn
from flax.training.common_utils import onehot
from functools import partial
import jax.numpy as jnp


def mae_loss(pred, target, mask, norm_pix_loss: bool = False):
    if norm_pix_loss:
        mean = target.mean(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True)
        target = (target - mean) / (var + 1e-6) ** .5
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss.mean(axis=-1)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    print("loss.shape:", loss.shape)
    return loss
