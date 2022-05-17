import os
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import SACLearner
from continuous_control.datasets import ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'quadruped-run', 'Environment name.')
flags.DEFINE_string('save_dir', './out/', 'Logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_integer('reset_interval', int(2e5), 'Periodicity of resets.')
flags.DEFINE_boolean('resets', False, 'Periodically reset the agent networks.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))

    kwargs = dict(FLAGS.config)
    assert kwargs.pop('algo') == 'sac'
    updates_per_step = kwargs.pop('updates_per_step')
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    agent = SACLearner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], **kwargs)
    
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if i >= FLAGS.start_training:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                agent.update(batch)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            
        if FLAGS.resets and i % FLAGS.reset_interval == 0:
            # create a completely new agent
            agent = SACLearner(FLAGS.seed + i,
                               env.observation_space.sample()[np.newaxis],
                               env.action_space.sample()[np.newaxis], **kwargs)


if __name__ == '__main__':
    app.run(main)
