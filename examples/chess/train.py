# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

import flax
from flax.training import train_state
import optax
from flax.training import orbax_utils
from flax.jax_utils import unreplicate
import orbax.checkpoint

from typing import Any

from network import AZNet
from nn_utils import train_step, forward

# import os
# os.environ["WANDB_MODE"] = "disabled"

# In alphazero, there is little difference between standard MCTS algorithm.
# In standard MCTS, we have 4 steps: selection, expansion, simulation and backpropogation.
# In alphazero, we replace the simulation step with a neural network, which directly outputs the value estimation.
# Here, change the terminology in alphzero, we call selection + expansion as one simulation step.
# Then, here is only 3 steps: simulation, expansion (with NN evaluation) and backpropogation in alphzero.
class Config(BaseModel):
    env_id: pgx.EnvId = "chess"
    seed: int = 0 # For chess game, seed doesn't matter, we always start from empty board
    # network params
    num_filters: int = 128
    num_residual_blocks: int = 16
    # selfplay params
    selfplay_batch_size: int = 32 # This is also the total number of games, train once after one batch of games complete.
    num_simulations: int = 32 # number of simulations per move (one simulation is a loop of simulation, expansion (with NN evaluation) and backpropogation).
    max_num_steps: int = 256 # each game can have at most 256 steps to play.
    # training params
    training_batch_size: int = 4096 # after MCTS, we have selfplay_batch_size * max_num_steps samples to train on.
    learning_rate: float = 0.0001
    # eval params
    eval_interval: int = 8

    class Config:
        extra = "forbid"

conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

# Initialize environment
env = pgx.make(config.env_id)
# Currently there is no baseline for chess in pgx library.
# baseline = pgx.make_baseline_model(config.env_id + "_v0") # use a strong baseline for eval

# To use mctx library, we need to define 2 functions: root_fn and recurrent_fn.
# define recurrent_fn first.
def recurrent_fn(train_state, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value) = forward(train_state, state.observation)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state

# define a class to record selfplay output, each SelfplayOutput instance corresponds to one step in one game,
# Thus, after complete a batch of games, we have selfplay_batch_size * max_num_steps SelfplayOutput and each
# represents a training sample.
class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray

@jax.pmap
def selfplay(train_state, rng_key: jnp.ndarray) -> SelfplayOutput:
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value) = forward(train_state, state.observation)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=train_state,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray

@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )

@jax.pmap
def evaluate(rng_key, train_state):
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        (my_logits, _) = forward(train_state, state.observation)
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R

if __name__ == "__main__":
    wandb.init(project="chess-az", config=config.model_dump())
    # define neural network
    az_net = AZNet(num_actions=4672)  # 4672 is the number of possible moves in chess
    # Initialize network, warm-up.
    rng_key, subkey = jax.random.split(jax.random.PRNGKey(config.seed), 2)
    dummy_input = jnp.zeros((32, 8, 8, 119))
    model_variables = az_net.init(subkey, dummy_input, is_training=True)
    model_state, params = flax.core.pop(model_variables, "params")
    # define train state
    class TrainState(train_state.TrainState):
        model_state: Any
        metrics: dict
    train_state = TrainState.create(
        apply_fn=az_net.apply,
        params=params,
        model_state=model_state,
        tx=optax.adamw(config.learning_rate),
        metrics={},
    )
    print(az_net.tabulate(subkey, dummy_input, is_training=True))
    # detect devices
    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"Found {num_devices} devices: {devices}, putting model on devices.")
    train_state = jax.device_put_replicated(train_state, devices)

    # main training stage
    iter, hours, frames = 0, 0.0, 0
    log = {"iteration": iter, "hours": hours, "frames": frames}
    print(log)
    wandb.log(log)
    print('training start at ', datetime.datetime.now())
    while True:
        if iter % config.eval_interval == 0:
            # evaluate the model
            # Evaluation
            # rng_key, subkey = jax.random.split(rng_key)
            # keys = jax.random.split(subkey, num_devices)
            # R = evaluate(keys, train_state)
            # log.update(
            #     {
            #         f"eval/vs_baseline/avg_R": R.mean().item(),
            #         f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
            #         f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
            #         f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
            #     }
            # )

            # # store checkpoints
            # save_path = f'/home/zhiyang/projects/class_projects/MARL/pgx/examples/chess/saved_params/AZ_chess_{iter:06d}'
            # # unreplicate the whole TrainState
            # train_state_to_save = unreplicate(train_state)
            # ckpt = {
            #     'params': train_state_to_save.params,
            #     'model_state': train_state_to_save.model_state,
            #     'opt_state': train_state_to_save.opt_state
            #     }
            # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            # orbax_checkpointer.save(save_path, ckpt)
            # del train_state_to_save # relaese manually for long run.
            pass

        start_time = time.time()
        # run selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data = selfplay(train_state, keys) # data shape: (num_devices, max_num_steps, selfplay_batch_size // num_devices, ...)
        # Now, we have collected a batch of data, but data['reward'] is the immediate reward.
        # We need to compute the n-step return for each step.
        samples = compute_loss_input(data)
        # shuffle and batch the samples for training
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )
        # train the model
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            train_state, train_info = train_step(train_state, minibatch)
            policy_losses.append(train_info['policy_loss'].mean().item())
            value_losses.append(train_info['value_loss'].mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        end_time = time.time()
        iter += 1
        hours += (end_time - start_time) / 3600.0
        # print and log the training info
        log = {
                "iteration": iter,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        print(log)
        wandb.log(log)


