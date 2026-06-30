"""PPO (Proximal Policy Optimization) with Flax NNX.

Implements clipped PPO objective with GAE (Generalized Advantage
Estimation) for actor-critic training. Designed for integration
with the SafePolicy wrapper (Actor + Diff-QP).
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax


class ActorCritic(nnx.Module):
    """Joint Actor-Critic network.

    Actor: Gaussian policy π(a|s) = N(μ(s), σ²I)
    Critic: V(s) scalar value estimate
    """

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int = 64,
                 rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.backbone = nnx.Sequential(
            nnx.Linear(n_obs, hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.tanh,
        )
        self.actor_head = nnx.Linear(hidden_dim, n_act, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros((n_act,)))
        self.critic_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        features = self.backbone(x)
        mean = self.actor_head(features)
        log_std = self.log_std[...]
        value = self.critic_head(features).squeeze()
        return mean, log_std, value

    def get_action(self, x: jnp.ndarray, key: jnp.ndarray):
        """Sample action from the policy."""
        mean, log_std, value = self(x)
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(key, mean.shape)
        log_prob = _gaussian_log_prob(action, mean, std)
        return action, log_prob, value

    def evaluate_actions(self, x: jnp.ndarray, actions: jnp.ndarray):
        """Evaluate log-prob and value for given actions (for PPO update)."""
        mean, log_std, value = self(x)
        std = jnp.exp(log_std)
        log_prob = _gaussian_log_prob(actions, mean, std)
        return log_prob, value


def _gaussian_log_prob(action, mean, std):
    """Log probability of action under diagonal Gaussian."""
    z = (action - mean) / std
    return -0.5 * jnp.sum(z ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))


def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray,
                dones: jnp.ndarray, gamma: float = 0.99,
                lam: float = 0.95) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns.

    Parameters
    ----------
    rewards : (T,) rewards at each step
    values : (T,) value estimates
    dones : (T,) done flags (1.0 = terminal)
    gamma : discount factor
    lam : GAE lambda

    Returns
    -------
    advantages : (T,)
    returns : (T,)
    """
    rev_rewards = jnp.flip(rewards)
    rev_values = jnp.flip(values)
    rev_dones = jnp.flip(dones)

    def _scan_step(carry, inputs):
        gae = carry
        r, v, d_next, v_next = inputs
        delta = r + gamma * v_next * (1.0 - d_next) - v
        gae = delta + gamma * lam * (1.0 - d_next) * gae
        return gae, gae

    rev_next_values = jnp.concatenate([rev_values[1:], jnp.array([0.0])])
    rev_next_dones = jnp.concatenate([rev_dones[1:], jnp.array([1.0])])

    _, rev_advantages = jax.lax.scan(
        _scan_step,
        0.0,
        (rev_rewards, rev_values, rev_next_dones, rev_next_values),
    )

    advantages = jnp.flip(rev_advantages)
    returns = advantages + values
    return advantages, returns


def ppo_clip_loss(ratio: jnp.ndarray, advantages: jnp.ndarray,
                   clip_eps: float = 0.2) -> jnp.ndarray:
    """PPO clipped surrogate loss (to be minimized).

    L = -E[min(ratio * A, clip(ratio, 1±ε) * A)]
    """
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
    return loss


class PPOTrainer:
    """PPO training loop manager.

    Handles:
    - Computing GAE advantages
    - Multiple epochs of PPO updates
    """

    def __init__(self, model: ActorCritic, lr: float = 3e-4,
                 clip_eps: float = 0.2, gamma: float = 0.99,
                 lam: float = 0.95, epochs: int = 10,
                 minibatch_size: int = 64, entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        self.model = model
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(nnx.state(model))

    def train_step(self, batch: dict):
        """One PPO update epoch on a batch of trajectories.

        batch keys: 'obs', 'actions', 'old_log_probs', 'advantages', 'returns'
        """
        graphdef, state = nnx.split(self.model)

        def loss_fn(state):
            model = nnx.merge(graphdef, state)
            log_probs, values = model.evaluate_actions(
                batch['obs'], batch['actions'])
            values = values.squeeze()

            # Policy loss
            ratio = jnp.exp(log_probs - batch['old_log_probs'])
            adv = (batch['advantages'] - jnp.mean(batch['advantages'])) / \
                  (jnp.std(batch['advantages']) + 1e-8)
            policy_loss = ppo_clip_loss(ratio, adv, self.clip_eps)

            # Value loss
            value_loss = jnp.mean((values - batch['returns']) ** 2)

            # Entropy bonus
            mean, log_std, _ = model(batch['obs'])
            entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
            entropy_loss = -self.entropy_coef * entropy

            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
            return total_loss

        loss_val, grads = jax.value_and_grad(loss_fn)(state)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        state = optax.apply_updates(state, updates)
        nnx.update(self.model, state)
        return loss_val
