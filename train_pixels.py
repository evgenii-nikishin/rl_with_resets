import os
import random

import jax
import flax
import numpy as np
import jax.numpy as jnp
import optax
import tqdm
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import DrQLearner
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
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_integer('reset_interval', 25000, 'Periodicity of resets.')
flags.DEFINE_boolean('resets', False, 'Periodically reset last actor / critic layers.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/drq.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)

    gray_scale = kwargs.pop('gray_scale')
    image_size = kwargs.pop('image_size')

    def make_pixel_env(seed, video_folder):
        return make_env(FLAGS.env_name,
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=gray_scale)

    env = make_pixel_env(FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    assert kwargs.pop('algo') == 'drq'
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    
    obs_demo = env.observation_space.sample()
    action_demo = env.action_space.sample()
    agent = DrQLearner(FLAGS.seed,
                       obs_demo[np.newaxis],
                       action_demo[np.newaxis], **kwargs)
    
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)
    
    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
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
            # shared enc params: 388416
            # critic head(s) params: 366232
            # actor head params: 286882
            # so we reset roughtly half of the agent (both layer and param wise)
            
            # save encoder parameters
            old_critic_enc = agent.critic.params['SharedEncoder']
            # target critic has its own copy of encoder
            old_target_critic_enc = agent.target_critic.params['SharedEncoder']
            # save encoder optimizer statistics
            old_critic_enc_opt = agent.critic.opt_state_enc
            
            # create new agent: note that the temperature is new as well
            agent = DrQLearner(FLAGS.seed + i,
                               env.observation_space.sample()[np.newaxis],
                               env.action_space.sample()[np.newaxis], **kwargs)
            
            # resetting critic: copy encoder parameters and optimizer statistics
            new_critic_params = agent.critic.params.copy(
                add_or_replace={'SharedEncoder': old_critic_enc})
            agent.critic = agent.critic.replace(params=new_critic_params, 
                                                opt_state_enc=old_critic_enc_opt)
            
            # resetting actor: actor in DrQ uses critic's encoder
            # note we could have copied enc optimizer here but actor does not affect enc
            new_actor_params = agent.actor.params.copy(
                add_or_replace={'SharedEncoder': old_critic_enc})
            agent.actor = agent.actor.replace(params=new_actor_params)
            
            # resetting target critic
            new_target_critic_params = agent.target_critic.params.copy(
                add_or_replace={'SharedEncoder': old_target_critic_enc})
            agent.target_critic = agent.target_critic.replace(
                params=new_target_critic_params)


if __name__ == '__main__':
    app.run(main)
