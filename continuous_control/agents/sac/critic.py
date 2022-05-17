from typing import Tuple

import jax
import jax.numpy as jnp

from continuous_control.datasets import Batch
from continuous_control.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           soft_critic: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if soft_critic:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # q1, q2 = critic.apply({'params': critic_params}, batch.observations,
        #                       batch.actions)
        critic_fn = lambda actions: critic.apply({'params': critic_params}, 
                                                 batch.observations, actions)
        def _critic_fn(actions):
            q1, q2 = critic_fn(actions)
            return 0.5*(q1 + q2).mean(), (q1, q2)

        (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, 
                                                        has_aux=True)(
                                                            batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
            'critic_pnorm': tree_norm(critic_params),
            'critic_agnorm': jnp.sqrt((action_grad ** 2).sum(-1)).mean(0)
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info['critic_gnorm'] = info.pop('grad_norm')

    return new_critic, info
