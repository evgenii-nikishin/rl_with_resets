"""
 Copyright 2022, The Primacy Bias in RL Authors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""

"""Compact implementation of the full Rainbow agent in Jax.

Specifically, we implement the following components from Rainbow:

    * n-step updates
    * prioritized replay
    * distributional RL
    * double_dqn
    * noisy
    * dueling

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

import collections
import copy
import functools
import time

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent as dopamine_rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import optax
import flax
import numpy as onp
import tensorflow as tf
from discrete_control import networks
from discrete_control.replay_memory import batched_buffer as tdrbs
from collections import OrderedDict
from flax.core.frozen_dict import FrozenDict

def copy_within_frozen_tree(old, new, prefix):
    new_entry = old[prefix].copy(add_or_replace=new)
    return old.copy(add_or_replace={prefix: new_entry})

def copy_params(old, new, keys=("encoder", "transition_model")):
    if isinstance(old, dict) or isinstance(old, OrderedDict) or isinstance(old, FrozenDict):
        fresh_dict = {}
        for k, v in old.items():
            if k in keys:
                fresh_dict[k] = v
            else:
                fresh_dict[k] = copy_params(old[k], new[k], keys)
        return fresh_dict
    else:
        return new

@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_leaves(tree)))

@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, support):
    """Select an action from the set of available actions."""
    epsilon = jnp.where(
        eval_mode, epsilon_eval,
        epsilon_fn(epsilon_decay_period, training_steps, min_replay_history,
                   epsilon_train))

    def q_online(state, key, actions=None, do_rollout=False):
        return network_def.apply(params, state, actions=actions,
                                 do_rollout=do_rollout, key=key,
                                 support=support, mutable=["batch_stats"])

    rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
    p = jax.random.uniform(rng1, shape=(state.shape[0],))
    q_values = get_q_values_no_actions(q_online, state, rng2)
    # q_values = network_def.apply(params, state, key=rng2, eval_mode=eval_mode,
    #                              support=support).q_values

    best_actions = jnp.argmax(q_values, axis=-1)
    new_actions = jnp.where(p <= epsilon,
        jax.random.randint(rng3, (state.shape[0],), 0, num_actions,),
        best_actions)
    return rng, new_actions

@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name="batch")
def get_q_values_no_actions(model, states, rng):
    results = model(states, actions=None, do_rollout=False, key=rng)[0]
    return results.q_values

@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, None), axis_name="batch")
def get_logits(model, states, actions, do_rollout, rng):
    results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
    return results.logits, results.latent


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, None), axis_name="batch")
def get_q_values(model, states, actions, do_rollout, rng):
    results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
    return results.q_values, results.latent


@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name="batch")
def get_spr_targets(model, states, key):
    results = model(states, key)
    return results


@functools.partial(jax.jit, static_argnums=(0, 3, 13, 14, 15, 17))
def train(network_def, online_params, target_params, optimizer, optimizer_state, states, actions, next_states,
          rewards, terminals, same_traj_mask, loss_weights, support,
          cumulative_gamma, double_dqn, distributional, rng, spr_weight):
  """Run a training step."""

  current_state = states[:, 0]
  online_params = online_params
  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  use_spr = spr_weight > 0

  def q_online(state, key, actions=None, do_rollout=False):
    return network_def.apply(
        online_params,
        state,
        actions=actions,
        do_rollout=do_rollout,
        key=key,
        support=support,
        mutable=["batch_stats"])

  def q_target(state, key):
    return network_def.apply(
        target_params, state, key=key, support=support, mutable=["batch_stats"])

  def encode_project(state, key):
    latent, _ = network_def.apply(
        target_params,
        state,
        method=network_def.encode,
        mutable=["batch_stats"])
    latent = latent.reshape(-1)
    return network_def.apply(
        target_params,
        latent,
        key=key,
        eval_mode=True,
        method=network_def.project)

  def loss_fn(params, target, spr_targets, loss_multipliers):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_online(state, key, actions=None, do_rollout=False):
      return network_def.apply(
          params,
          state,
          actions=actions,
          do_rollout=do_rollout,
          key=key,
          support=support,
          mutable=["batch_stats"])

    if distributional:
      (logits, spr_predictions) = get_logits(q_online, current_state,
                                             actions[:, :-1], use_spr, rng)
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions[:, 0])
      dqn_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits)
      td_error = dqn_loss + jnp.nan_to_num(target*jnp.log(target)).sum(-1)
    else:
      q_values, spr_predictions = get_q_values(q_online, current_state,
                                               actions[:, :-1], use_spr, rng)
      q_values = jnp.squeeze(q_values)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions[:, 0])
      dqn_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)
      td_error = dqn_loss

    if use_spr:
      spr_predictions = spr_predictions.transpose(1, 0, 2)
      spr_predictions = spr_predictions / jnp.linalg.norm(
          spr_predictions, 2, -1, keepdims=True)
      spr_targets = spr_targets / jnp.linalg.norm(
          spr_targets, 2, -1, keepdims=True)
      spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
      spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)
    else:
      spr_loss = 0

    loss = dqn_loss + spr_weight * spr_loss

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, (loss, dqn_loss, spr_loss, td_error)

  # Use the weighted mean loss for gradient computation.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_output(q_online, q_target, next_states, rewards, terminals,
                         support, cumulative_gamma, double_dqn, distributional,
                         rng1)

  if use_spr:
    future_states = states[:, 1:]
    spr_targets = get_spr_targets(
        encode_project, future_states.reshape(-1, *future_states.shape[2:]),
        rng1)
    spr_targets = spr_targets.reshape(*future_states.shape[:2],
                                      *spr_targets.shape[1:]).transpose(
                                          1, 0, 2)
  else:
    spr_targets = None

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, (loss, dqn_loss, spr_loss, td_error)), grad =\
      grad_fn(online_params, target, spr_targets, loss_weights)
  grad_norm = tree_norm(grad)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, mean_loss, dqn_loss, spr_loss, \
         grad_norm, td_error, rng2


@functools.partial(
    jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None, None, None), axis_name="batch")
def target_output(model, target_network, next_states, rewards, terminals,
                  support, cumulative_gamma, double_dqn, distributional, rng):
    """Builds the C51 target distribution or DQN target Q-values."""
    is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
    # Incorporate terminal state to discount factor.
    gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

    target_network_dist, _ = target_network(next_states, key=rng)
    if double_dqn:
        # Use the current network for the action selection
        next_state_target_outputs, _ = model(next_states, key=rng)
    else:
        next_state_target_outputs = target_network_dist
    # Action selection using Q-values for next-state
    q_values = jnp.squeeze(next_state_target_outputs.q_values)
    next_qt_argmax = jnp.argmax(q_values)

    if distributional:
        # Compute the target Q-value distribution
        probabilities = jnp.squeeze(target_network_dist.probabilities)
        next_probabilities = probabilities[next_qt_argmax]
        target_support = rewards + gamma_with_terminal * support
        target = dopamine_rainbow_agent.project_distribution(target_support, next_probabilities, support)
    else:
        # Compute the target Q-value
        next_q_values = jnp.squeeze(target_network_dist.q_values)
        replay_next_qt_max = next_q_values[next_qt_argmax]
        target = rewards + gamma_with_terminal * replay_next_qt_max

    return jax.lax.stop_gradient(target)

@gin.configurable
def create_optimizer(name='adam', learning_rate=6.25e-5, beta1=0.9, beta2=0.999,
                     eps=1.5e-4, centered=False, warmup=0):
  """Create an optimizer for training.

  Currently, only the Adam and RMSProp optimizers are supported.

  Args:
    name: str, name of the optimizer to create.
    learning_rate: float, learning rate to use in the optimizer.
    beta1: float, beta1 parameter for the optimizer.
    beta2: float, beta2 parameter for the optimizer.
    eps: float, epsilon parameter for the optimizer.
    centered: bool, centered parameter for RMSProp.
    warmup: int, warmup steps for learning rate.

  Returns:
    A flax optimizer.
  """
  if name == 'adam':
    logging.info('Creating Adam optimizer with settings lr=%f, beta1=%f, '
                 'beta2=%f, eps=%f', learning_rate, beta1, beta2, eps)
    if warmup == 0:
        return optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)
    return optax.inject_hyperparams(optax.adam)(learning_rate=optax.linear_schedule(0, learning_rate, warmup), b1=beta1, b2=beta2, eps=eps)
  elif name == 'rmsprop':
    logging.info('Creating RMSProp optimizer with settings lr=%f, beta2=%f, '
                 'eps=%f', learning_rate, beta2, eps)
    if warmup == 0:
        return optax.rmsprop(learning_rate, decay=beta2, eps=eps,
                         centered=centered)
    return optax.inject_hyperparams(optax.rmsprop)(learning_rate=optax.linear_schedule(0, learning_rate, warmup), decay=beta2, eps=eps,
                         centered=centered)
  else:
    raise ValueError('Unsupported optimizer {}'.format(name))


@gin.configurable
class JaxSPRAgent(dqn_agent.JaxDQNAgent):
    """A compact implementation of the full Rainbow agent."""

    def __init__(self,
                 num_actions,
                 noisy=False,
                 dueling=False,
                 double_dqn=False,
                 distributional=True,
                 data_augmentation=False,
                 network=networks.RainbowDQNNetwork,
                 num_atoms=51,
                 vmax=10.,
                 vmin=None,
                 jumps=5,
                 spr_weight=5,
                 batch_size=32,
                 replay_ratio=64,
                 log_every=100,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 replay_scheme='prioritized',
                 replay_type='deterministic',
                 reset_every=-1,
                 reset_offset=0,
                 total_resets=0,
                 encoder_warmup=0,
                 head_warmup=0,
                 reset_head=False,
                 reset_projection=False,
                 reset_encoder=False,
                 reset_noise=False,
                 updates_on_reset=0,
                 summary_writer=None,
                 seed=None):
        """Initializes the agent and constructs the necessary components.

        Args:
            num_actions: int, number of actions the agent can take at any state.
            noisy: bool, Whether to use noisy networks or not.
            dueling: bool, Whether to use dueling network architecture or not.
            double_dqn: bool, Whether to use Double DQN or not.
            distributional: bool, whether to use distributional RL or not.
            data_augmentation: bool, Whether to use data augmentation or not.
            network: flax.linen Module, neural network used by the agent initialized
                by shape in _create_network below. See
                dopamine.jax.networks.RainbowNetwork as an example.
            num_atoms: int, the number of buckets of the value function distribution.
            vmax: float, the value distribution support is [vmin, vmax].
            vmin: float, the value distribution support is [vmin, vmax]. If vmin is
                None, it is set to -vmax.
            jumps: int, how many steps to use for SPR. 5 is original.
            spr_weight: float, what weight to give SPR loss. 5 is original.
            batch_size: int, batch size for training.
            log_every: int, how often to log
            epsilon_fn: function expecting 4 parameters: (decay_period, step,
                warmup_steps, epsilon). This function should return the epsilon value
                used for exploration during training.
            replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
                replay memory.
            replay_type: str, 'deterministic' or 'regular', specifies the type of
                replay buffer to create.
            reset_every: int, how often to reset.
            reset_offset: int, offsets reset period.
            total_resets: int, how many resets to perform during training.
            encoder_warmup: int, LR warmup steps for encoder after resets.
            head_warmup: int, LR warmup steps for head after resets.
            reset_head: bool, whether or not to reset final layer
            reset_projection: bool, reset penultimate layer
            reset_encoder: bool, reset CNN encoder
            reset_noise: bool, reset noisy nets parameters in head.
            updates_on_reset: int, how many offline updates to perform after
                each reset.
            summary_writer: SummaryWriter object, for outputting training statistics.
            seed: int, a seed for Jax RNG and initialization.
        """
        logging.info('Creating %s agent with the following parameters:',
                     self.__class__.__name__)
        logging.info('\t double_dqn: %s', double_dqn)
        logging.info('\t noisy_networks: %s', noisy)
        logging.info('\t dueling_dqn: %s', dueling)
        logging.info('\t distributional: %s', distributional)
        logging.info('\t data_augmentation: %s', data_augmentation)
        logging.info('\t replay_scheme: %s', replay_scheme)
        logging.info('\t total_resets: %s', total_resets)
        logging.info('\t reset_every: %s', reset_every)
        logging.info('\t reset_encoder: %s', reset_encoder)
        logging.info('\t reset_noise: %s', reset_noise)
        logging.info('\t reset_projection: %s', reset_projection)
        logging.info('\t updates_on_reset: %s', updates_on_reset)
        # We need this because some tools convert round floats into ints.
        vmax = float(vmax)
        self._num_atoms = num_atoms
        vmin = vmin if vmin else -vmax
        self._support = jnp.linspace(vmin, vmax, num_atoms)
        self._replay_scheme = replay_scheme
        self._replay_type = replay_type
        self._double_dqn = double_dqn
        self._noisy = noisy
        self._dueling = dueling
        self._distributional = distributional
        self._data_augmentation = data_augmentation
        self._replay_ratio = replay_ratio
        self._batch_size = batch_size
        self._jumps = jumps
        self.spr_weight = spr_weight
        self.log_every = log_every

        self.reset_every = int(reset_every)
        self.reset_offset = int(reset_offset)
        self.reset_head = reset_head
        self.reset_projection = reset_projection
        self.reset_encoder = reset_encoder
        self.reset_noise = reset_noise
        self.updates_on_reset = int(updates_on_reset)
        self.remaining_resets = int(total_resets)

        self.encoder_warmup = int(encoder_warmup)
        self.head_warmup = int(head_warmup)

        self.replay_elements = None

        super().__init__(
            num_actions=num_actions,
            network=functools.partial(
                network, num_atoms=num_atoms,
                noisy=self._noisy,
                dueling=self._dueling,
                distributional=self._distributional),
            epsilon_fn=epsilon_fn,
            summary_writer=summary_writer,
            seed=seed)

    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        self.state_shape = self.state.shape
        self.online_params = self.network_def.init(rng, x=self.state,
                                                   actions=jnp.zeros((5,)),
                                                   do_rollout=self.spr_weight>0,
                                                   support=self._support)
        optimizer = create_optimizer(self._optimizer_name, warmup=self.head_warmup)
        encoder_optimizer = create_optimizer(self._optimizer_name, warmup=self.encoder_warmup)

        self.encoder_mask = FrozenDict(
            {"params": {"encoder": True, "transition_model": True,
                        "head": False, "projection": False, "predictor": False}}
        )
        self.head_mask = FrozenDict(
            {"params": {"encoder": False, "transition_model": False,
                        "head": True, "projection": True, "predictor": True}}
        )

        self.optimizer = optax.chain(
            optax.masked(encoder_optimizer, self.encoder_mask),
            optax.masked(optimizer, self.head_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        if self._replay_scheme not in ['uniform', 'prioritized']:
            raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
        if self._replay_type not in ['deterministic']:
            raise ValueError('Invalid replay type: {}'.format(self._replay_type))
        if self._replay_scheme == "prioritized":
            buffer = tdrbs.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
                observation_shape=self.observation_shape,
                stack_size=self.stack_size,
                update_horizon=self.update_horizon,
                gamma=self.gamma,
                subseq_len=self._jumps + 1,
                observation_dtype=self.observation_dtype,)
        else:
            buffer = tdrbs.JaxSubsequenceParallelEnvReplayBuffer(
                observation_shape=self.observation_shape,
                stack_size=self.stack_size,
                update_horizon=self.update_horizon,
                gamma=self.gamma,
                subseq_len=self._jumps + 1,
                observation_dtype=self.observation_dtype,)

        self._batch_size = buffer._batch_size
        self.n_envs = buffer.n_envs
        self.start = time.time()

        print("Operating with {} environments, batch size {} and replay ratio {}".format(self.n_envs, self._batch_size,
                                                                                         self._replay_ratio))
        self._num_updates_per_train_step = self._replay_ratio*self.n_envs // self._batch_size
        print("Calculated {} updates per step".format(self._num_updates_per_train_step))

        print("Setting min_replay_history to {} from {}".format(self.min_replay_history/self.n_envs,
                                                                self.min_replay_history))
        print("Setting epsilon_decay_period to {} from {}".format(self.epsilon_decay_period/self.n_envs,
                                                                  self.epsilon_decay_period))
        self.min_replay_history = (self.min_replay_history/self.n_envs)
        self.epsilon_decay_period = (self.epsilon_decay_period/self.n_envs)

        return buffer

    def _sample_from_replay_buffer(self):
        self._rng, rng = jax.random.split(self._rng)
        samples = self._replay.sample_transition_batch(rng)
        types = self._replay.get_transition_elements()
        self.replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            self.replay_elements[element_type.name] = element

    def reset_weights(self):
        if self.remaining_resets <= 0:
            print("All resets completed, skipping")
            return
        else:
            self.remaining_resets -= 1
            print("Resetting weights; {} resets remaining".format(self.remaining_resets))

        self._rng, rng = jax.random.split(self._rng)
        if len(self.state_shape) < len(self.state.shape):
            state = self.state[0].reshape(*self.state_shape)
        else:
            state = self.state.reshape(*self.state_shape)
        online_network_params = self.network_def.init(rng, x=state,
                                                      actions=jnp.zeros((5,)),
                                                      do_rollout=self.spr_weight > 0,
                                                      support=self._support)
        optim_state = self.optimizer.init(self.online_params)

        keys_to_copy = []
        if not self.reset_projection:
            keys_to_copy += ["projection", "predictor"]
        if not self.reset_encoder:
            keys_to_copy += ["encoder", "transition_model"]
        if not self.reset_noise:
            keys_to_copy += ["kernell", "biass"]

        updated_optim_state = []
        for i in range(len(optim_state)):
            optim_to_copy = copy_params(dict(self.optimizer_state[i]._asdict()),
                                        dict(optim_state[i]._asdict()),
                                        keys=keys_to_copy)
            optim_to_copy = FrozenDict(optim_to_copy)
            updated_optim_state.append(optim_state[i]._replace(**optim_to_copy))

        self.optimizer_state = tuple(updated_optim_state)
        self.online_params = FrozenDict(copy_params(self.online_params,
                                                    online_network_params,
                                                    keys=keys_to_copy))

        print("Running {} steps after reset".format(self.updates_on_reset))

        for i in range(self.updates_on_reset):
            self._train_step()

    def preprocess_states(self):
        self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
        self.replay_elements['state'] = networks.process_inputs(
            self.replay_elements['state'],
            rng=rng1,
            data_augmentation=self._data_augmentation)
        self.replay_elements['next_state'] = networks.process_inputs(
            self.replay_elements['next_state'][:, 0],
            rng=rng2,
            data_augmentation=self._data_augmentation)

    def _training_step_update(self, step_index=0):
        """Gradient update during every training step."""
        # print("Inter-batch time {}".format(time.time() - self.start))
        interbatch_time = time.time() - self.start
        self.start = time.time()
        train_start = time.time()
        if self.replay_elements is None:
            self._sample_from_replay_buffer()
            self.preprocess_states()
        # print("Sampling took {}".format(time.time() - train_start))
        sampling_time = time.time() - train_start
        train_start = time.time()

        aug_time = time.time() - train_start
        train_start = time.time()

        if self._replay_scheme == 'prioritized':
            # The original prioritized experience replay uses a linear exponent
            # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
            # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
            # suggested a fixed exponent actually performs better, except on Pong.
            probs = self.replay_elements['sampling_probabilities']
            probs = probs
            # Weight the loss by the inverse priorities.
            loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
            loss_weights /= jnp.max(loss_weights)
        else:
            # Uniform weights if not using prioritized replay.
            loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, loss, mean_loss, \
        dqn_loss, spr_loss, grad_norm, td_errors, \
        self._rng = train(
            self.network_def, self.online_params, self.target_network_params,
            self.optimizer, self.optimizer_state,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'][:, 0],
            self.replay_elements['terminal'][:, 0],
            self.replay_elements['same_trajectory'][:, 1:], loss_weights,
            self._support, self.cumulative_gamma, self._double_dqn,
            self._distributional, self._rng, self.spr_weight
        )
        # Jax will happily run train asynchronously unless we block on its
        # output, so we can sample a new batch while it's running.
        self._sample_from_replay_buffer()
        self.preprocess_states()

        if self._replay_scheme == 'prioritized':
            # Rainbow and prioritized replay are parametrized by an exponent
            # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
            # leave it as is here, using the more direct sqrt(). Taking the square
            # root "makes sense", as we are dealing with a squared loss.  Add a
            # small nonzero value to the loss to avoid 0 priority items. While
            # technically this may be okay, setting all items to 0 priority will
            # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
            self._replay.set_priority(self.replay_elements['indices'],
                                      jnp.sqrt(dqn_loss + 1e-10))
        #
        # print("Training took {}".format(time.time() - train_start))
        training_time = time.time() - train_start
        if self.training_steps % self.log_every == 0 and step_index == 0:
            if self.summary_writer is not None:
                summary = tf.compat.v1.Summary(value=[
                    tf.compat.v1.Summary.Value(
                        tag='TotalLoss', simple_value=float(mean_loss)),
                    tf.compat.v1.Summary.Value(
                        tag='DQNLoss', simple_value=float(dqn_loss.mean())),
                    tf.compat.v1.Summary.Value(
                        tag='GradNorm', simple_value=float(grad_norm)),
                    tf.compat.v1.Summary.Value(
                        tag='PNorm', simple_value=float(tree_norm(self.online_params))),
                    tf.compat.v1.Summary.Value(
                        tag='TD Error', simple_value=float(td_errors.mean())),
                    tf.compat.v1.Summary.Value(
                        tag='SPRLoss', simple_value=float(spr_loss.mean())),
                    tf.compat.v1.Summary.Value(
                        tag='Inter-batch time', simple_value=float(interbatch_time)),
                    tf.compat.v1.Summary.Value(
                        tag='Sampling time', simple_value=float(sampling_time)),
                    tf.compat.v1.Summary.Value(
                        tag='Augmentation time', simple_value=float(aug_time)),
                    tf.compat.v1.Summary.Value(
                        tag='Training time', simple_value=float(training_time)),
                ])
                self.summary_writer.add_summary(summary, self.training_steps)

    def _store_transition(self,
                          last_observation,
                          action,
                          reward,
                          is_terminal,
                          *args,
                          priority=None,
                          episode_end=False):
        """Stores a transition when in training mode."""
        is_prioritized = (
                isinstance(self._replay,
                           prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer)
                or isinstance(self._replay,
                              tdrbs.PrioritizedJaxSubsequenceParallelEnvReplayBuffer))
        if is_prioritized and priority is None:
            priority = onp.ones((last_observation.shape[0]))
            if self._replay_scheme == 'uniform':
                pass  # Already 1, doesn't matter
            else:
                priority.fill(self._replay.sum_tree.max_recorded_priority)

        if not self.eval_mode:
            self._replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                *args,
                priority=priority,
                episode_end=episode_end)

    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
            (1) A minimum number of frames have been added to the replay buffer.
            (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_network_params to target_network_params if
        training steps is a multiple of target update period.
        """
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                for i in range(self._num_updates_per_train_step):
                    self._training_step_update(i)

            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

            if (self.training_steps + 1) % (self.reset_every + self.updates_on_reset) == self.reset_offset\
                    and self.reset_every > 0:
                print("Resetting weights at {}".format(self.training_steps))
                self.reset_weights()

        self.training_steps += 1

    def _reset_state(self, n_envs):
        """Resets the agent state by filling it with zeros."""
        self.state = onp.zeros(n_envs, *self.state_shape)

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
          observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        observation = observation.squeeze(-1)
        if len(observation.shape) == len(self.observation_shape):
            self._observation = onp.reshape(observation, self.observation_shape)
        else:
            self._observation = onp.reshape(observation,
                                            (observation.shape[0],
                                             *self.observation_shape))
        # Swap out the oldest frame with the current frame.
        self.state = onp.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self._observation

    def reset_all(self, new_obs):
        """Resets the agent state by filling it with zeros."""
        n_envs = new_obs.shape[0]
        self.state = onp.zeros((n_envs, *self.state_shape))
        self._record_observation(new_obs)

    def reset_one(self, env_id):
        self.state[env_id].fill(0)

    def delete_one(self, env_id):
        self.state = onp.concatenate([self.state[:env_id],
                                      self.state[env_id+1:]], 0)

    def cache_train_state(self):
        self.training_state = (copy.deepcopy(self.state),
                               copy.deepcopy(self._last_observation),
                               copy.deepcopy(self._observation))

    def restore_train_state(self):
        self.state,\
        self._last_observation,\
        self._observation = self.training_state

    def log_transition(self, observation, action, reward, terminal, episode_end):
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, action, reward, terminal,
                                   episode_end=episode_end)

    def step(self):
        """Records the most recent transition, returns the agent's next action, and trains if appropriate."""
        if not self.eval_mode:
            self._train_step()
        state = networks.process_inputs(self.state, data_augmentation=False)
        self._rng, self.action = select_action(
            self.network_def, self.online_params, state, self._rng,
            self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
            self.epsilon_decay_period, self.training_steps, self.min_replay_history,
            self.epsilon_fn, self._support)
        self.action = onp.asarray(self.action)
        return self.action
