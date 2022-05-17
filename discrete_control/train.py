# coding=utf-8
# Copyright 2021 The Atari 100k Precipice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Entry point for Atari 100k experiments.

"""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train
import numpy as np
import tensorflow.compat.v2 as tf
from discrete_control import eval_run_experiment
from discrete_control.agents import rainbow_agent
import gin


FLAGS = flags.FLAGS
CONFIGS_DIR = './configs'
AGENTS = ['rainbow', 'der', 'dopamine_der', 'DrQ', 'OTRainbow', "SPR"]

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'rainbow', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_integer('agent_seed', None, 'If None, use the run_number')
flags.DEFINE_string(
        'load_replay_dir', None, 'Directory to load the initial replay buffer from '
        'a fixed dataset. If None, no transitions are loaded. ')
flags.DEFINE_string(
        'tag', None, 'Tag for this run')
flags.DEFINE_boolean(
        'save_replay', False, 'Whether to save agent\'s final replay buffer '
        'as a fixed dataset to ${base_dir}/replay_logs.')
flags.DEFINE_integer(
        'load_replay_number', None, 'Index of the replay run to load the initial '
        'replay buffer from a fixed dataset. If None, uses the `run_number`.')
flags.DEFINE_boolean('max_episode_eval', True,
                                         'Whether to use `MaxEpisodeEvalRunner` or not.')
flags.DEFINE_boolean('wandb', False, 'Also log to wandb.')

def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)

def init_wandb(base_dir, seed, tag=None, agent=None):
    os.environ['WANDB_MODE'] = 'offline'
    os.makedirs(base_dir, exist_ok=True)
    import wandb
    from gin import config
    clean_cfg = {k[1]: v for k, v in config._CONFIG.items()}
    clean_cfg["seed"] = seed
    clean_cfg["tag"] = tag
    clean_cfg["agent"] = agent
    wandb.init(config=clean_cfg, sync_tensorboard=True, dir=base_dir)


def create_load_replay_dir(xm_params):
    """Creates the directory for loading fixed replay data."""
    problem_name, run_number = '', ''
    for param, value in xm_params.items():
        if param.endswith('game_name'):
            problem_name = value
        elif param.endswith('run_number'):
            run_number = str(value)
    replay_dir = FLAGS.load_replay_dir
    if replay_dir:
        if FLAGS.load_replay_number:
            replay_number = str(FLAGS.load_replay_number)
        else:
            replay_number = run_number
        replay_dir = os.path.join(replay_dir, problem_name, replay_number,
                                                            'replay_logs')
    return replay_dir


def create_agent(sess, environment,
                       seed,
                       summary_writer=None):
    """Helper function for creating agent."""
    return rainbow_agent.JaxSPRAgent(
            num_actions=environment.action_space.n,
            seed=seed,
            summary_writer=summary_writer)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    logging.info('Setting random seed: %d', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def main(unused_argv):
    """Main method.

    Args:
        unused_argv: Arguments (unused).
    """
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.disable_v2_behavior()

    base_dir = FLAGS.base_dir
    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    # Add code for setting random seed using the run_number
    set_random_seed(FLAGS.run_number)
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    if FLAGS.wandb:
        init_wandb(base_dir, FLAGS.run_number, FLAGS.tag, FLAGS.agent)
    # Set the Jax agent seed
    seed = FLAGS.run_number if not FLAGS.agent_seed else FLAGS.agent_seed
    create_agent_fn = functools.partial(
            create_agent, seed=seed)
    if FLAGS.max_episode_eval:
        runner_fn = eval_run_experiment.DataEfficientAtariRunner
        logging.info('Using MaxEpisodeEvalRunner for evaluation.')
        kwargs = {}    # No additional flags should be passed.
        runner = runner_fn(base_dir, create_agent_fn, **kwargs)
    else:
        runner = run_experiment.Runner(base_dir, create_agent_fn)
    runner.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
