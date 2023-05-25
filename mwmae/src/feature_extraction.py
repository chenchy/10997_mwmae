import pickle
import glob
import json
import functools
import os
import math
import soundfile as sf
import tqdm
from absl import logging
from flax import optim
from flax.training import checkpoints
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from functools import partial
from .training_utils.trainstate import TrainState_v2
from .training_utils import training_utilities, metrics_helper

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
from .training_utils.training_utilities import TrainingMode
from . import mae_utilities
