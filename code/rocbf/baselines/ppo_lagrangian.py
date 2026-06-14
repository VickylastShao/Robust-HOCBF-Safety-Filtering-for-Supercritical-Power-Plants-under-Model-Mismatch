"""PPO-Lagrangian: soft constraint enforcement via dual gradient descent.

Extends PPOTrainer with an auto-tuned Lagrange multiplier λ that
penalizes constraint violations: L = L_policy + λ * L_cost - η * log(λ).

Cost is computed from barrier function values: cost = max(0, -min_i h_i(x)).
λ is updated via dual ascent: λ ← max(0, λ + lr_λ * (cost - limit)).
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae, ppo_clip_loss


class PPOTrainerLagrangian(PPOTrainer):
    """PPO with Lagrangian multiplier for soft constraint enforcement.

    Adds cost terms: L = L_policy + λ * L_cost
    where L_cost = mean(max(0, cost - cost_limit)).
    λ is auto-tuned via dual gradient descent.
    """

    def __init__(self, model: ActorCritic, lr: float = 3e-4,
                 cost_limit: float = 0.0,
                 lagrangian_lr: float = 0.01,
                 **kwargs):
        super().__init__(model, lr=lr, **kwargs)
        self.cost_limit = cost_limit
        self.lagrangian_lr = lagrangian_lr
        self.lambda_cost = 1.0

    def train_step(self, batch: dict):
        """One PPO-Lagrangian update epoch.

        batch keys: 'obs', 'actions', 'old_log_probs', 'advantages',
                    'returns', 'costs'
        """
        graphdef, state = nnx.split(self.model)

        def loss_fn(state):
            model = nnx.merge(graphdef, state)
            log_probs, values = model.evaluate_actions(
                batch['obs'], batch['actions'])
            values = values.squeeze()

            # Policy loss (standard PPO clip)
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

            # Lagrangian cost loss
            costs = batch.get('costs', None)
            if costs is not None:
                cost_loss = jnp.mean(jnp.maximum(0.0, costs - self.cost_limit))
            else:
                cost_loss = jnp.array(0.0)

            total_loss = (policy_loss + self.value_coef * value_loss
                          + entropy_loss + self.lambda_cost * cost_loss)
            return total_loss

        loss_val, grads = jax.value_and_grad(loss_fn)(state)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        state = optax.apply_updates(state, updates)
        nnx.update(self.model, state)

        # Update Lagrange multiplier via dual ascent
        if 'costs' in batch:
            mean_cost = float(jnp.mean(batch['costs']))
            self.lambda_cost = max(
                0.0, self.lambda_cost + self.lagrangian_lr * (mean_cost - self.cost_limit))
            self.lambda_cost = min(self.lambda_cost, 100.0)

        return loss_val


def compute_step_costs(constraint_vals_list: list) -> jnp.ndarray:
    """Compute per-step cost from barrier function values.

    cost_t = max(0, -min_i h_i(x_t))

    Parameters
    ----------
    constraint_vals_list : list of dict
        Each dict has constraint name → h value from constraint.check_all()

    Returns
    -------
    costs : (T,) cost at each step
    """
    costs = []
    for vals in constraint_vals_list:
        min_h = min(float(v) for v in vals.values())
        costs.append(max(0.0, -min_h))
    return jnp.array(costs)
