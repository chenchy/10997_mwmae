import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial
from typing import Callable, Optional, Any

Dtype = Any


class LearnableFrontendWrapper(nn.Module):
    model_cls: Callable
    frontend_cls: Optional[Callable] = None
    spec_aug: Optional[Callable] = None
    dtype: Dtype = jnp.float32
    features_only: bool = False
    invert_features: bool = False

    def setup(self):
        if self.frontend_cls is not None:
            self.frontend = self.frontend_cls(dtype=jnp.float32, name="frontend")
        else:
            self.frontend = None
        self.model = self.model_cls(name="MaskedAutoencoderViT_0")
        if self.invert_features and self.frontend is not None:
            kernel_size = int(self.frontend.sample_rate * self.frontend.window_len // 1000 + 1)
            stride = int(self.frontend.sample_rate * self.frontend.window_stride // 1000)
            self.invert_feats = nn.ConvTranspose(1, kernel_size=(kernel_size,), strides=(stride,), padding="SAME")
    # @nn.compact
    def __call__(self, inputs, gumbel_temperature=None, train: bool = True):
        outputs = inputs
        print("in LearnableFrontendWrapper, shape:", outputs.shape)
        if self.frontend_cls is not None:
            outputs = self.frontend(outputs)
            outputs = outputs[Ellipsis, jnp.newaxis]
            outputs = outputs.astype(self.dtype)
            if self.spec_aug is not None and train:
                spec_aug_rng = self.make_rng('spec_aug')
                outputs, _ = self.spec_aug(outputs, spec_aug_rng)
        print("in LearnableFrontendWrapper, shape:", outputs.shape)
        if self.features_only:
            return self.model.forward_features(outputs, train=train)
        else:
            return self.model(outputs, train=train)

    def forward_features(self, inputs, train: bool=True):
        outputs = inputs
        if self.frontend_cls is not None:
            outputs = self.frontend(outputs)
            outputs = outputs[Ellipsis, jnp.newaxis]
            outputs = outputs.astype(self.dtype)
            if self.spec_aug is not None and train:
                spec_aug_rng = self.make_rng('spec_aug')
                outputs, _ = self.spec_aug(outputs, spec_aug_rng)
        return self.model.forward_features(outputs, train=train)
