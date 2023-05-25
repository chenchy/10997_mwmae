import os
import json
import logging
import ml_collections
import optax
from typing import Any
import jax
from jax import lax, random
import flax
from flax import jax_utils
from jax import numpy as jnp
from flax.training import checkpoints
from flax import optim
from flax.core import freeze, unfreeze
from .trainstate import TrainState_v2


DEFAULT_AUX_RNG_KEYS = ["dropout", "drop_path", "mixup", "spec_aug"]


def get_dtype(half_precision: bool):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_dtype


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x:x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def save_best_checkpoint(state, workdir, best_acc):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, prefix="best_ckpt_", keep=3)


def initialize(key, inp_shape, model, 
               aux_rng_keys=DEFAULT_AUX_RNG_KEYS):
    input_shape = (2,) + inp_shape

    @jax.jit
    def init(*args):
        return model.init(*args)
    num_keys = len(aux_rng_keys)
    key, *subkeys = jax.random.split(key, num_keys+1)
    rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}
    variables = init({'params': key, **rng_keys}, 
                     jnp.ones(input_shape, model.dtype))
    rngs = flax.core.FrozenDict(rng_keys)
    has_batch_stats = False
    has_buffers = False
    if "batch_stats" in variables.keys():
        has_batch_stats = True
    if "buffers" in variables.keys():
        has_buffers = True
    res = (
        variables['params'],
        variables['batch_stats'] if has_batch_stats else flax.core.freeze({}),
        variables['buffers'] if has_buffers else flax.core.freeze({}),
        rngs
    )
    #     return variables['params'], variables['batch_stats'], rngs

    # else:
    #     return variables['params'], flax.core.freeze({}), rngs
    return res

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if len(state.batch_stats) != 0:
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    else:
        return state


def step_lr(base_lr, reduce_every_n_steps, total_steps, alpha=0.5):
    schedules = []
    boundaries = []
    curr_lr = base_lr
    for step in range(1, total_steps+1, reduce_every_n_steps):
        boundaries.append(step-1)
        schedules.append(optax.constant_schedule(curr_lr))
        curr_lr *= alpha
    boundaries = boundaries[1:]
    schedule_fn = optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries
    )
    return schedule_fn


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int,
        num_epochs: int=None):
    """Create learning rate schedule."""
    if config.opt.schedule == "warmupcosine":
        logging.info("Using cosine learning rate decay")
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=base_learning_rate,
            transition_steps=config.opt.warmup_epochs * steps_per_epoch)
        if num_epochs is None:
            num_epochs = config.num_epochs
        cosine_epochs = max(num_epochs - config.opt.warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[config.opt.warmup_epochs * steps_per_epoch])
    elif config.opt.schedule == "cosine_decay":
        cosine_epochs = num_epochs
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = cosine_fn
    elif config.opt.schedule == "exp_decay":
        warmup_epochs = int(config.opt.get("warmup_epochs", 0))
        transition_steps = (num_epochs - warmup_epochs) * steps_per_epoch
        fixed_steps = warmup_epochs * steps_per_epoch
        schedule_fn = optax.exponential_decay(
            init_value=base_learning_rate,
            decay_rate=config.opt.get("decay_rate", 0.1),
            transition_begin=fixed_steps+1,
            transition_steps=transition_steps,
            end_value=config.opt.get("end_value", 1e-7)
        )
    elif config.opt.schedule == "step":
        lr_step_every = int(config.opt.get("step_epochs", 10) * steps_per_epoch)
        alpha = config.opt.get("alpha", 0.5)
        total_steps = steps_per_epoch * num_epochs
        schedule_fn = step_lr(base_learning_rate, 
                              reduce_every_n_steps=lr_step_every,
                              total_steps=total_steps,
                              alpha=alpha)
    else:
        schedule_fn = base_learning_rate
    return schedule_fn


def create_optimizer(config: ml_collections.ConfigDict, learning_rate_fn):
    optimizer_name = config.opt.optimizer.lower()
    if optimizer_name == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "lars":
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.opt.weight_decay
        )
    elif optimizer_name == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.opt.get("momentum", 0.9),
            nesterov=config.opt.get("nesterov", False)
        )
    else:
        raise ValueError("optimizer {} not supported. Valid values are [adamw, lars, sgd]")
    return tx


def create_train_state_from_pretrained(rng, config: ml_collections.ConfigDict,
                                       model, learning_rate_fn,
                                       pretrained_work_dir, 
                                       pretrained_prefix="checkpoint_",
                                       copy_all=False,
                                       to_copy=['encoder'],
                                       fc_only=False,
                                       additional_aux_rngs=[],
                                       apply_fn_override=None):
    logging.info("Making train state from pretrained..")
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None
    if additional_aux_rngs is not None and len(additional_aux_rngs) != 0:
        aux_rngs = DEFAULT_AUX_RNG_KEYS + additional_aux_rngs
    else:
        aux_rngs = DEFAULT_AUX_RNG_KEYS
    params, batch_stats, buffers, rng_keys = initialize(rng, config.input_shape, model, aux_rng_keys=aux_rngs)

    # load pretrained ckpt to a dictionary
    pretrained_state_dict = checkpoints.restore_checkpoint(pretrained_work_dir, None,
                                                           prefix=pretrained_prefix)
    pretrained_params = pretrained_state_dict['params']
    pretrained_batch_stats = pretrained_state_dict['batch_stats']
    pretrained_buffers = pretrained_state_dict['buffers']

    # unfreeze classifier params and batch_stats
    params = unfreeze(params)
    batch_stats = unfreeze(batch_stats)
    buffers = unfreeze(buffers)
    if copy_all:
        logging.info("copy_all is True. Attempting to copy all parameters by name")
        to_copy = list(set(params.keys()).intersection(set(pretrained_params.keys())))
    logging.info("Copying the following parameters: \n[{}]".format(",".join(to_copy)))
    # copy stuff
    for k in to_copy:
        assert k in params.keys() #and k in batch_stats.keys()
        params[k] = pretrained_params[k]
        try:
            batch_stats[k] = pretrained_batch_stats[k]
        except KeyError as ex:
            pass
        try:
            buffers[k] = pretrained_buffers[k]
        except KeyError as ex:
            pass

    # filter params based on fc_only argument
    if fc_only:
        logging.info("Finetuning fc-only layer")
        frozen_params = {}
        trainable_params = {}
        for k in params.keys():
            if k in to_copy:
                frozen_params[k] = params[k]
            else:
                trainable_params[k] = params[k]
    else:
        frozen_params = {}
        trainable_params = params

    # freeze classifier params and batch_stats
    trainable_params = freeze(trainable_params)
    frozen_params = freeze(frozen_params)
    batch_stats = freeze(batch_stats)
    buffers = freeze(buffers)

    # make the train state now
    tx = create_optimizer(config, learning_rate_fn)
    grad_accum_steps = config.opt.get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logging.info("Using gradient accumulation of {} steps".format(grad_accum_steps))
        tx = optax.MultiSteps(tx, grad_accum_steps)
    if apply_fn_override is not None:
        state = TrainState_v2.create(
        apply_fn=apply_fn_override,
        params=trainable_params,
        frozen_params=frozen_params,
        tx=tx,
        batch_stats=batch_stats,
        buffers=buffers,
        aux_rng_keys=rng_keys,
        dynamic_scale=dynamic_scale)
    else:    
        state = TrainState_v2.create(
            apply_fn=model.apply,
            params=trainable_params,
            frozen_params=frozen_params,
            tx=tx,
            batch_stats=batch_stats,
            buffers=buffers,
            aux_rng_keys=rng_keys,
            dynamic_scale=dynamic_scale)
    return state
