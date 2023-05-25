import os
import sys
from hear_api.runtime_v2 import RuntimeMAE
import ml_collections
import pathlib
base_fld = str(pathlib.Path(__file__).parent.resolve().parent.parent.joinpath("pretrained_weights"))
MW_MAE_MODEL_DIR = os.environ.get("MW_MAE_MODEL_DIR", base_fld)
print("loading pretrained checkpoints from base_fld:", MW_MAE_MODEL_DIR)


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    
    config.model = ml_collections.ConfigDict()
    config.model.arch = "vit_msm_base_200"
    config.model.type = "multiclass"                    # for parsing compatibility
    config.model.num_classes = 527                    # for parsing compatibility

    config.model.model_args = {
        "patch_size": (4, 16),
    }
    config.model.patch_embed_args = ml_collections.ConfigDict()

    config.model.pretrained = ""
    config.model.pretrained_fc_only = True
    config.model.patch_embed_args = ml_collections.ConfigDict()

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adamw"
    config.opt.learning_rate = 1.5e-4
    config.opt.weight_decay = 0.005
    config.opt.schedule = "warmupcosine"
    config.opt.warmup_epochs = 10
    config.opt.momentum = 0.9
    config.opt.norm_pix_loss = True
    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.batch_size = 8*128
    config.half_precision = False
    config.input_shape = (200, 80, 1)
    config.num_epochs = 100
    config.device = None

    return config


RUN_ID = 1
model_path = os.path.join(MW_MAE_MODEL_DIR, f"audioset/mae_base_200_16x4_384d-8h-4l_default_8x128_0.8_run{RUN_ID}")


def load_model(model_path=model_path, config=get_config()):
    model = RuntimeMAE(config=config, weights_dir=model_path)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
