import numpy as np
import jax
import jax.numpy as jnp
from jax import dtypes
import collections.abc
from itertools import repeat
from typing import Tuple, Optional


def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_1tuple = _ntuple(1)


def constant_init(key, shape, dtype=jnp.float_, constant=0.04):
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * constant


def _sample_negative_indices(features_shape: Tuple, num_negatives: int, attention_mask: Optional[np.ndarray] = None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length, hidden_size = features_shape
    if sequence_length <= 1:
        raise ValueError(
            "`features should have `sequence_length` > 1, but are of shape "
            f"(batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
        )

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = []
    for batch_idx in range(batch_size):
        high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
        sampled_indices_slice = np.random.randint(0, high, size=(num_negatives * sequence_length,))
        sampled_negative_indices.append(sampled_indices_slice)

    sampled_negative_indices = np.asarray(sampled_negative_indices, dtype=np.int32)

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    feature_indices = np.broadcast_to(np.arange(sequence_length)[:, None], (sequence_length, num_negatives)).flatten()

    # avoid sampling the same positive vector, but keep the distribution uniform
    sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

    # correct for batch size
    for batch_idx in range(1, batch_size):
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices
