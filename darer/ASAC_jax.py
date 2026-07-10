"""Pure-JAX re-implementation of ASAC (see ASAC.py for the torch/sb3 original).

Networks, Adam, gradient clipping and polyak averaging are implemented directly
on JAX pytrees (no flax/optax dependency). The env loop, replay buffer and
logging are reused from BaseAgent.
"""
from functools import partial
from typing import Any, NamedTuple, Optional
import pickle
import sys

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import gymnasium as gym
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim

sys.path.append('darer')
from BaseAgent import BaseAgent
from utils import logger_at_folder

# Match SB3's SquashedDiagGaussian bounds
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6

AGGREGATORS = {
    'min': lambda q: jnp.min(q, axis=0),
    'max': lambda q: jnp.max(q, axis=0),
    'mean': lambda q: jnp.mean(q, axis=0),
}


# ---------------------------------------------------------------------------
# Networks (params are plain pytrees of dicts/lists)
# ---------------------------------------------------------------------------
def linear_init(key, in_dim, out_dim):
    # Same distribution as torch.nn.Linear's default init
    w_key, b_key = jax.random.split(key)
    bound = 1.0 / np.sqrt(in_dim)
    return {
        'w': jax.random.uniform(w_key, (in_dim, out_dim), jnp.float32, -bound, bound),
        'b': jax.random.uniform(b_key, (out_dim,), jnp.float32, -bound, bound),
    }


def critic_init(key, nS, nA, hidden_dim):
    dims = [nS + nA, hidden_dim, hidden_dim, 1]
    keys = jax.random.split(key, len(dims) - 1)
    return [linear_init(k, i, o) for k, i, o in zip(keys, dims[:-1], dims[1:])]


def critic_apply(params, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    x = jax.nn.relu(x @ params[0]['w'] + params[0]['b'])
    x = jax.nn.relu(x @ params[1]['w'] + params[1]['b'])
    return x @ params[2]['w'] + params[2]['b']


# Ensemble of critics with params stacked along a leading axis
critics_apply = jax.vmap(critic_apply, in_axes=(0, None, None))


def actor_init(key, nS, nA, hidden_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        'latent': [linear_init(k1, nS, hidden_dim),
                   linear_init(k2, hidden_dim, hidden_dim)],
        'mu': linear_init(k3, hidden_dim, nA),
        'log_std': linear_init(k4, hidden_dim, nA),
    }


def actor_apply(params, obs):
    x = obs
    for layer in params['latent']:
        x = jax.nn.relu(x @ layer['w'] + layer['b'])
    mu = x @ params['mu']['w'] + params['mu']['b']
    log_std = x @ params['log_std']['w'] + params['log_std']['b']
    return mu, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def actor_sample(params, obs, key):
    """Sample a squashed-gaussian action in [-1, 1] and its log-prob."""
    mu, log_std = actor_apply(params, obs)
    std = jnp.exp(log_std)
    gaussian = mu + std * jax.random.normal(key, mu.shape)
    action = jnp.tanh(gaussian)
    log_prob = jnp.sum(
        -0.5 * jnp.square((gaussian - mu) / std) - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1)
    log_prob -= jnp.sum(jnp.log(1 - jnp.square(action) + EPS), axis=-1)
    return action, log_prob


# ---------------------------------------------------------------------------
# Optimizer / pytree utilities
# ---------------------------------------------------------------------------
class AdamState(NamedTuple):
    mu: Any
    nu: Any
    count: jnp.ndarray


def adam_init(params):
    return AdamState(jax.tree.map(jnp.zeros_like, params),
                     jax.tree.map(jnp.zeros_like, params),
                     jnp.zeros((), jnp.int32))


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    count = state.count + 1
    mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, grads)
    nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, grads)
    mu_c = 1 - b1 ** count
    nu_c = 1 - b2 ** count
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / mu_c) / (jnp.sqrt(v / nu_c) + eps),
        params, mu, nu)
    return params, AdamState(mu, nu, count)


def clip_grads(grads, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)


def clip_grads_per_net(grads, max_norm):
    """Clip each critic of the stacked ensemble by its own global norm."""
    sq = jax.tree.map(lambda g: jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))),
                      grads)
    norms = jnp.sqrt(sum(jax.tree.leaves(sq)))
    scale = jnp.minimum(1.0, max_norm / (norms + 1e-6))
    return jax.tree.map(
        lambda g: g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), grads)


def max_abs_grad(grads):
    return jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in jax.tree.leaves(grads)]))


@jax.jit
def polyak_update(online, target, tau):
    return jax.tree.map(lambda o, t: tau * o + (1 - tau) * t, online, target)


class TrainState(NamedTuple):
    actor_params: Any
    actor_opt: AdamState
    critic_params: Any
    critic_opt: AdamState
    target_params: Any
    log_ent_coef: jnp.ndarray
    ent_opt: AdamState
    theta: jnp.ndarray
    penalty: jnp.ndarray


def make_train_step(nS, nA, aggregator, learn_ent_coef, ent_coef_value,
                    target_entropy, logpi0, use_dones, tau_theta,
                    learning_rate, actor_learning_rate, ent_learning_rate,
                    max_grad_norm):
    agg = AGGREGATORS[aggregator]

    def train_step(ts: TrainState, states, actions, next_states, dones, rewards, key):
        pi_key, next_key = jax.random.split(key)

        # Action by the current actor for the sampled state
        _, log_prob = actor_sample(ts.actor_params, states, pi_key)
        log_prob = log_prob.reshape(-1, 1)

        if learn_ent_coef:
            # ent_coef is read before the temperature update, as in the original
            ent_coef = jnp.exp(ts.log_ent_coef)
            ent_loss_fn = lambda lec: -(lec * (log_prob + target_entropy)).mean()
            ent_coef_loss, ent_grad = jax.value_and_grad(ent_loss_fn)(ts.log_ent_coef)
            log_ent_coef, ent_opt = adam_step(ts.log_ent_coef, ent_grad,
                                              ts.ent_opt, ent_learning_rate)
        else:
            ent_coef = jnp.asarray(ent_coef_value)
            log_ent_coef, ent_opt = ts.log_ent_coef, ts.ent_opt
            ent_coef_loss = jnp.asarray(0.0)

        # --- Critic targets (no gradient flows through this block) ---
        next_actions, next_log_prob = actor_sample(ts.actor_params, next_states, next_key)
        next_q_values = agg(critics_apply(ts.target_params, next_states, next_actions))
        next_v_values = next_q_values - ent_coef * (next_log_prob.reshape(-1, 1) - logpi0)

        penalty = ts.penalty
        if use_dones:
            # make the penalty same as mean of non-terminating rewards:
            raw_penalty = 20 * jnp.max(
                jnp.take_along_axis(rewards, dones.astype(jnp.int32), axis=0))
            penalty = raw_penalty * tau_theta + (1 - tau_theta) * ts.penalty
            next_v_values = next_v_values * (1 - dones) - penalty * dones

        new_theta = jnp.mean(rewards - ent_coef * (log_prob - logpi0))
        # always subtract the mean of target q values to try and keep it
        # centered (in terms of its span):
        baseline = jnp.mean(critics_apply(ts.target_params,
                                          jnp.zeros((1, nS)), jnp.zeros((1, nA))))
        next_v_values = next_v_values - baseline
        target_q_values = jax.lax.stop_gradient(rewards - ts.theta + next_v_values)

        theta = ts.theta * (1 - tau_theta) + tau_theta * new_theta

        # --- Critic update ---
        def critic_loss_fn(cp):
            qs = critics_apply(cp, states, actions)  # (num_nets, B, 1)
            return 0.5 * jnp.sum(jnp.mean(jnp.square(qs - target_q_values[None]),
                                          axis=(1, 2)))

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(ts.critic_params)
        if max_grad_norm is not None:
            critic_grads = clip_grads_per_net(critic_grads, max_grad_norm)
        critic_params, critic_opt = adam_step(ts.critic_params, critic_grads,
                                              ts.critic_opt, learning_rate)

        # --- Actor update (uses the freshly updated critics, same noise) ---
        def actor_loss_fn(ap):
            actions_pi, log_prob_pi = actor_sample(ap, states, pi_key)
            q_values_pi = agg(critics_apply(critic_params, states, actions_pi))
            return (ent_coef * log_prob_pi.reshape(-1, 1) - q_values_pi).mean()

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(ts.actor_params)
        if max_grad_norm is not None:
            actor_grads = clip_grads(actor_grads, max_grad_norm)
        actor_params, actor_opt = adam_step(ts.actor_params, actor_grads,
                                            ts.actor_opt, actor_learning_rate)

        new_ts = TrainState(actor_params, actor_opt, critic_params, critic_opt,
                            ts.target_params, log_ent_coef, ent_opt, theta, penalty)
        metrics = {
            'ent_coef': ent_coef.reshape(()),
            'ent_coef_loss': ent_coef_loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'mean_q': next_q_values.mean(),
            'penalty': penalty,
            'baseline': baseline,
            'new_theta': new_theta,
            'next_logprob': next_log_prob.mean(),
            'mean_reward': rewards.mean(),
            'max_grad_norm': max_abs_grad(critic_grads),
            'actor_max_grad': max_abs_grad(actor_grads),
            'temp': ent_coef.reshape(()),
        }
        return new_ts, metrics

    return jax.jit(train_step)


class ASACJax(BaseAgent):
    def __init__(self,
                 *args,
                 policy: str = 'MlpPolicy',
                 actor_learning_rate: Optional[float] = None,
                 use_ppi: bool = False,
                 use_dones: bool = True,
                 jax_seed: int = 0,
                 name_suffix: str = '',
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.kwargs.update(locals())
        self.kwargs.pop('self')
        self.kwargs.pop('args')
        self.kwargs.pop('kwargs')
        self.kwargs.pop('__class__')
        self.algo_name = 'ASACJax' + '-no' * (not use_dones) + \
            '-auto' * (self.beta == 'auto') + name_suffix
        self.use_dones = use_dones
        self.use_ppi = use_ppi
        self.actor_learning_rate = self.learning_rate if actor_learning_rate is None \
            else actor_learning_rate
        self.nA = get_action_dim(self.env.action_space)
        self.nS = get_flattened_obs_dim(self.env.observation_space)

        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self.logpi0 = float(np.log(1 / self.nA))
        if self.beta != 'auto':
            self.ent_coef = self.beta ** (-1)
        else:
            self.ent_coef = 'auto'
        self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        self.key = jax.random.PRNGKey(jax_seed)
        self._initialize_networks()

    def _initialize_networks(self):
        self.key, actor_key, *critic_keys = jax.random.split(self.key, 2 + self.num_nets)
        actor_params = actor_init(actor_key, self.nS, self.nA, self.hidden_dim)
        critic_params = jax.tree.map(
            lambda *nets: jnp.stack(nets),
            *[critic_init(k, self.nS, self.nA, self.hidden_dim) for k in critic_keys])

        learn_ent_coef = isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto")
        if learn_ent_coef:
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            # Optimize the log of the entropy coefficient, as in the original
            log_ent_coef = jnp.log(jnp.ones(1) * init_value)
            ent_coef_value = init_value
        else:
            log_ent_coef = jnp.zeros(1)
            ent_coef_value = float(self.ent_coef)

        self.train_state = TrainState(
            actor_params=actor_params,
            actor_opt=adam_init(actor_params),
            critic_params=critic_params,
            critic_opt=adam_init(critic_params),
            target_params=critic_params,
            log_ent_coef=log_ent_coef,
            ent_opt=adam_init(log_ent_coef),
            theta=jnp.asarray(0.0),
            penalty=jnp.asarray(0.0),
        )
        self._train_step = make_train_step(
            nS=self.nS, nA=self.nA, aggregator=self.aggregator,
            learn_ent_coef=learn_ent_coef, ent_coef_value=ent_coef_value,
            target_entropy=self.target_entropy, logpi0=self.logpi0,
            use_dones=self.use_dones, tau_theta=self.tau_theta,
            learning_rate=self.learning_rate,
            actor_learning_rate=self.actor_learning_rate,
            ent_learning_rate=1e-4, max_grad_norm=self.max_grad_norm)
        self._sample_fn = jax.jit(actor_sample)
        self._det_action_fn = jax.jit(lambda p, obs: jnp.tanh(actor_apply(p, obs)[0]))

    def exploration_policy(self, state):
        action, buffer_action = self._sample_action(state)
        return (action, buffer_action), 0

    def evaluation_policy(self, state):
        obs = np.asarray(state, dtype=np.float32).reshape(1, self.nS)
        squashed = np.asarray(self._det_action_fn(self.train_state.actor_params, obs))[0]
        return self._unscale_action(squashed).reshape(self.env.action_space.shape)

    def _scale_action(self, action):
        low, high = self.env.action_space.low, self.env.action_space.high
        return 2.0 * (action - low) / (high - low) - 1.0

    def _unscale_action(self, action):
        low, high = self.env.action_space.low, self.env.action_space.high
        return low + 0.5 * (action + 1.0) * (high - low)

    def _sample_action(self, state, n_envs=1):
        """
        Sample an action according to the exploration policy: random during
        warm-up, stochastic squashed-gaussian afterwards. Returns the action to
        take in the environment and the scaled ([-1, 1]) action for the buffer.
        """
        if self.env_steps < self.learning_starts:
            unscaled_action = np.array([self.env.action_space.sample()
                                        for _ in range(n_envs)]).squeeze(0)
        else:
            obs = np.asarray(state, dtype=np.float32).reshape(1, self.nS)
            self.key, sample_key = jax.random.split(self.key)
            squashed, _ = self._sample_fn(self.train_state.actor_params, obs, sample_key)
            unscaled_action = self._unscale_action(np.asarray(squashed)[0])

        if isinstance(self.env.action_space, gym.spaces.Box):
            # We store the scaled action in the buffer
            buffer_action = self._scale_action(unscaled_action)
            action = unscaled_action
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def gradient_descent(self, batch, grad_step):
        states, actions, next_states, dones, rewards = batch
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float32)

        self.key, step_key = jax.random.split(self.key)
        self.train_state, metrics = self._train_step(
            self.train_state, to_np(states), to_np(actions), to_np(next_states),
            to_np(dones), to_np(rewards), step_key)

        self.theta = self.train_state.theta
        self.lr = self.learning_rate
        self.logger.record("train/tau_theta", self.tau_theta)
        for name, value in metrics.items():
            self.logger.record(f"train/{name}", float(value))

    def _update_target(self):
        self.train_state = self.train_state._replace(
            target_params=polyak_update(self.train_state.critic_params,
                                        self.train_state.target_params,
                                        self.tau))

    def save(self, path=None):
        import inspect
        valid = set(inspect.signature(BaseAgent.__init__).parameters) | \
            set(inspect.signature(ASACJax.__init__).parameters)
        kwargs = {k: v for k, v in self.kwargs.items() if k in valid}
        kwargs.pop('env_id', None)
        if path is None:
            path = str(self)
        with open(path, 'wb') as f:
            pickle.dump({'kwargs': kwargs,
                         'train_state': jax.device_get(self.train_state),
                         'class': self.__class__.__name__}, f)

    @staticmethod
    def load(path, env_id, **new_kwargs):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        kwargs = state['kwargs']
        kwargs.update(new_kwargs)
        agent = ASACJax(env_id, **kwargs)
        agent.train_state = jax.device_put(state['train_state'])
        return agent


def main():
    env_id = 'InvertedPendulum-v5'
    with open(f'hparams/{env_id}/asac.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    agent = ASACJax(env_id, **params, device='cpu',
                    tensorboard_log=f'local-asac-jax-{env_id}',
                    render=False, log_interval=200,
                    save_best=True)
    agent.learn(total_timesteps=500_000)


if __name__ == '__main__':
    main()
