import jax
import numpy as np
import jax.numpy as jnp
import sys
sys.path.append("..")
import torch
from .feature_helper import LogMelSpec, get_timestamps
from src import mae_utilities
from src.training_utils import training_utilities
from functools import partial


def get_grid_size(img_size, patch_size):
    grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    return grid_size


def forward(batch, state, model):
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats,
        "buffers": state.buffers
    }
    logits = model.apply(
        variables, batch, train=False, mutable=False, method=model.forward_features
    )
    return logits



class RuntimeMAE(torch.nn.Module):
    def __init__(self, config, weights_dir):
        super().__init__()
        self.config = config
        self.local_batch_size = 1
        self.devices = [jax.local_devices()[0]]
        rng = jax.random.PRNGKey(0)
        patch_embed = mae_utilities.get_patch_embed(config)
        model_cls = mae_utilities.get_model_cls(config)
        frontend_cls = None
        model = mae_utilities.create_model(
            model_cls=model_cls, 
            patch_embed_projection_module=patch_embed,
            frontend_cls=frontend_cls,
            half_precision=config.half_precision,
            num_classes=config.model.num_classes,
            lin_probe=config.model.pretrained_fc_only
        )
        self.model = model
        print("loading pretrained weights from {}".format(weights_dir))
        learning_rate_fn = training_utilities.create_learning_rate_fn(config, 0.1, 100)
        state = training_utilities.create_train_state_from_pretrained(rng, config,
                                                                      self.model, learning_rate_fn,
                                                                      weights_dir,
                                                                      config.model.get("pretrained_prefix",
                                                                                    "checkpoint_"),
                                                                      copy_all=True,
                                                                      to_copy=[],
                                                                      fc_only=True,
                                                                      additional_aux_rngs=['random_masking'],
                                                                      apply_fn_override=self.model.forward_features)
        self.forward_jit = jax.jit(partial(forward, state=state, model=self.model), device=self.devices[0])
        self.state = state
        self.log_mel_spec = LogMelSpec()
        self.grid_size = get_grid_size(img_size=self.model.img_size, patch_size=self.model.patch_size)
        self.input_size = self.model.img_size
        self.embed_dim = self.model.embed_dim
        self.sample_rate = 16000
    
    def to_feature(self, batch_audio):
        x = self.log_mel_spec(batch_audio)
        mean = torch.mean(x, [1, 2], keepdims=True)
        std = torch.std(x, [1, 2], keepdims=True)

        x = (x - mean) / (std + 1e-8)

        x = x.permute(0, 2, 1)
        x = jnp.asarray(x.detach().cpu().numpy())
        x = x[Ellipsis, jnp.newaxis]
        return x

    # def normalize_batch(self, x):
    #     if self.mean is not None and self.std is not None:
    #         x = (x - self.mean) / self.std
    #     return x

    def get_features(self, batch_audio):
        x = self.to_feature(batch_audio)
        return x

    def encode(self, lms):
        x = lms

        patch_fbins = self.grid_size[1]
        unit_frames = self.input_size[0]

        embed_d = self.embed_dim

        cur_frames = x.shape[1]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            pad_arg = [(0, 0), (0, pad_frames), (0, 0), (0, 0)]
            x = jnp.pad(x, pad_arg, mode="reflect")

        embeddings = []
        for i in range(x.shape[1] // unit_frames):
            x_inp = x[:, i*unit_frames:(i+1)*unit_frames, Ellipsis]
            logits = self.forward_jit(x_inp)
            embeddings.append(logits)
        x = jnp.concatenate(embeddings, axis=1)
        pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
        if pad_emb_frames > 0:
            x = x[:, :-pad_emb_frames, Ellipsis]
        return x

    def audio2feats(self, audio):
        x = self.get_features(audio)
        x = self.encode(x)
        x = torch.from_numpy(np.array(x.to_py()))
        return x

    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)        
        x = torch.mean(x, dim=1)
        return x
    
    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
