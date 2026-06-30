"""Phase 3: Train PPO + Multi-Constraint Robust HOCBF on CCS.

Two-timescale protocol adapted for the 1000 MW USC CCS system:
- Fast timescale (RL): PPO update every episode
- Slow timescale (GP): GP residual update every M episodes

The CCS system has 3 states, 3 inputs, and 4 HOCBF constraints
(pressure high/low, enthalpy high/low) stacked via MultiConstraintRobustHOCBF.
"""
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintRobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.env import CCSEnv


SCENARIOS = [None, "heat_absorption", "pressure_oscillation", "coupled", "nonlinear"]
SCENARIO_LABELS = ["Nominal", "S1:Heat", "S2:Pressure", "S3:Coupled", "S4:Nonlinear"]


def collect_ccs_gp_data(dynamics, n_transitions: int = 500,
                        key: jnp.ndarray | None = None):
    """Collect GP training data from CCS random policy rollouts.

    Computes residuals Delta-f = (x' - x)/dt - f_nominal(x) - g(x)u
    which captures model mismatch from the bias-corrected nominal.
    """
    if key is None:
        key = jax.random.key(0)

    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)

    X_list, Y_list = [], []
    x = x0

    for _ in range(n_transitions):
        key, u_key = jax.random.split(key)
        # Random action near equilibrium
        u = u0 + jnp.array([
            jax.random.uniform(u_key, (), minval=-5.0, maxval=5.0),
            jax.random.uniform(u_key, (), minval=-20.0, maxval=20.0),
            jax.random.uniform(u_key, (), minval=-2.0, maxval=2.0),
        ])

        x_next = dynamics.step(x, u)

        # Residual: Delta-f = (x' - x)/dt - f_nominal(x) - g(x)u
        residual = (x_next - x) / dynamics.dt - dynamics.f_nominal(x) - (dynamics.g(x) @ u).squeeze()

        X_list.append(x)
        Y_list.append(residual)

        # Reset if out of bounds
        if jnp.any(jnp.abs(x_next[:3] - x0) > jnp.array([30.0, 5.0, 300.0])):
            key, reset_key = jax.random.split(key)
            x = x0 + jnp.array([5.0, 0.5, 50.0]) * jax.random.normal(reset_key, (3,))
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)


def make_multi_hocbf(dynamics, constraint, gp_residual, u_max=100.0):
    """Create MultiConstraintRobustHOCBF with all CCS constraints.

    Uses closed-loop drift f_cl = f_nominal + g*u0 for well-conditioned
    HOCBF psi-chain (avoids numerical explosion from open-loop drift ~10^6).
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    hocbf_list = []

    # Pressure high: relative degree 2
    hocbf_list.append(RobustHOCBF(
        h_fn=constraint.h_pressure_high,
        f_fn=dynamics.f_closed_loop,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[0.5, 0.5],
        gp_residual=gp_residual,
        u_max=u_max,
        u0=u0,
    ))

    # Pressure low: relative degree 2
    hocbf_list.append(RobustHOCBF(
        h_fn=constraint.h_pressure_low,
        f_fn=dynamics.f_closed_loop,
        g_fn=dynamics.g,
        relative_degree=2,
        k_gains=[0.5, 0.5],
        gp_residual=gp_residual,
        u_max=u_max,
        u0=u0,
    ))

    # Enthalpy high: relative degree 1
    hocbf_list.append(RobustHOCBF(
        h_fn=constraint.h_enthalpy_high,
        f_fn=dynamics.f_closed_loop,
        g_fn=dynamics.g,
        relative_degree=1,
        k_gains=[1.0],
        gp_residual=gp_residual,
        u_max=u_max,
        u0=u0,
    ))

    # Enthalpy low: relative degree 1
    hocbf_list.append(RobustHOCBF(
        h_fn=constraint.h_enthalpy_low,
        f_fn=dynamics.f_closed_loop,
        g_fn=dynamics.g,
        relative_degree=1,
        k_gains=[1.0],
        gp_residual=gp_residual,
        u_max=u_max,
        u0=u0,
    ))

    return MultiConstraintRobustHOCBF(hocbf_list)


def collect_rollout(model, dynamics, multi_hocbf, qp_solver, constraint,
                    x0, u0, key, n_steps=300):
    """Collect one episode of rollout data with multi-constraint safe policy."""
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': []}

    x = x0
    total_reward = 0.0
    violations = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        action, log_prob, value = model.get_action(x[:3], action_key)

        # Add equilibrium offset
        u_rl = action + u0

        # QP safety filter (deviation formulation: constraint on v = u - u0)
        A, b = multi_hocbf.qp_matrices(x[:3])
        v_rl = u_rl - u0  # deviation reference
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        u_safe = u0 + v_safe

        next_x = dynamics.step(x, u_safe)

        # Check constraints
        x_phys = next_x[:3] if dynamics.delay_order > 0 else next_x
        constraint_vals = constraint.check_all(x_phys, u_safe)
        terminated = any(v < 0 for v in constraint_vals.values())

        # Reward: tracking + effort
        y = dynamics.output(x_phys, u_safe)
        y0 = dynamics.output(x0, u0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum((u_safe - u0) ** 2)
        )
        if terminated:
            reward -= 100.0

        rollout['obs'].append(x[:3])  # Use physical state for policy
        rollout['actions'].append(u_safe - u0)  # Delta actions for training
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(terminated))

        if terminated:
            violations += 1

        total_reward += float(reward)
        x = next_x

        if terminated:
            break

    for k in rollout:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations


def train_phase3(n_episodes: int = 2000, n_steps: int = 300,
                 eval_every: int = 100, n_eval: int = 5,
                 gp_update_interval: int = 50,
                 n_pretrain: int = 3000,
                 load_ratio: float = 1.0,
                 delay_order: int = 0):
    """Train PPO + Multi-Constraint Robust HOCBF on CCS."""
    print("=== Phase 3: CCS Scenario Deployment Training ===\n")
    sys.stdout.flush()

    # --- Phase A: GP pre-training ---
    print("Phase A: GP pre-training on CCS residuals...")
    sys.stdout.flush()

    key = jax.random.key(42)
    X_all, Y_all = [], []

    for scenario in SCENARIOS:
        env = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=scenario)
        key, data_key = jax.random.split(key)
        X, Y = collect_ccs_gp_data(env, n_transitions=n_pretrain // len(SCENARIOS),
                                    key=data_key)
        X_all.append(X)
        Y_all.append(Y)

    X_combined = jnp.concatenate(X_all, axis=0)
    Y_combined = jnp.concatenate(Y_all, axis=0)

    gp = GPResidual(n_dims=3, noise_variance=1e-4)
    gp.fit(X_combined, Y_combined)
    print(f"  GP fitted on {gp.n_training_points} transitions, "
          f"beta = {GPResidual.compute_beta(3, gp.n_training_points):.2f}")
    sys.stdout.flush()

    # --- Phase B: PPO + Multi-Constraint Robust HOCBF ---
    print("\nPhase B: Two-timescale training...\n")
    sys.stdout.flush()

    dynamics = USCCSDynamics(delay_order=delay_order, load_ratio=load_ratio)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2650, 2850),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=dynamics)
    x0, u0 = dynamics.equilibrium(load_ratio)

    multi_hocbf = make_multi_hocbf(dynamics, constraint, gp, u_max=100.0)
    qp_solver = DifferentiableQP()

    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)

    key = jax.random.key(42)

    for ep in range(n_episodes):
        # Sample scenario
        key, scenario_key = jax.random.split(key)
        scenario_idx = jax.random.randint(scenario_key, (), 0, len(SCENARIOS))
        scenario = SCENARIOS[int(scenario_idx)]

        env = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=scenario)

        # Rebuild HOCBF with current GP
        multi_hocbf = make_multi_hocbf(dynamics, constraint, gp, u_max=100.0)

        key, rollout_key = jax.random.split(key)
        rollout, ep_reward, violations = collect_rollout(
            model, env, multi_hocbf, qp_solver, constraint,
            x0, u0, rollout_key, n_steps)

        if rollout['obs'].shape[0] < 2:
            continue

        # PPO update
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'])

        batch = {
            'obs': rollout['obs'],
            'actions': rollout['actions'],
            'old_log_probs': rollout['log_probs'],
            'advantages': advantages,
            'returns': returns,
        }

        for _ in range(trainer.epochs):
            loss = trainer.train_step(batch)

        # Slow timescale: GP update
        if (ep + 1) % gp_update_interval == 0:
            print(f"  [GP Update at ep {ep+1}] Refitting GP...")
            sys.stdout.flush()

            key, gp_key = jax.random.split(key)
            X_new_list, Y_new_list = [], []
            for scenario_gp in SCENARIOS:
                env_gp = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=scenario_gp)
                key, data_key = jax.random.split(gp_key)
                X_new, Y_new = collect_ccs_gp_data(
                    env_gp, n_transitions=200, key=data_key)
                X_new_list.append(X_new)
                Y_new_list.append(Y_new)

            X_new = jnp.concatenate(X_new_list, axis=0)
            Y_new = jnp.concatenate(Y_new_list, axis=0)
            gp.incremental_update(X_new, Y_new)

            multi_hocbf = make_multi_hocbf(dynamics, constraint, gp, u_max=100.0)

            print(f"    GP now has {gp.n_training_points} points, "
                  f"beta = {GPResidual.compute_beta(3, gp.n_training_points):.2f}")
            sys.stdout.flush()

        # Evaluation
        if (ep + 1) % eval_every == 0:
            print(f"Episode {ep+1}: ", end="")
            sys.stdout.flush()
            for i, scenario in enumerate(SCENARIOS):
                env_eval = UncertainUSCCSDynamics(
                    delay_order=delay_order, load_ratio=load_ratio,
                    uncertainty_scenario=scenario)
                hocbf_eval = make_multi_hocbf(dynamics, constraint, gp, u_max=100.0)
                key, eval_key = jax.random.split(key)

                total_viol = 0
                total_r = 0.0
                for _ in range(n_eval):
                    key, ep_key = jax.random.split(eval_key)
                    _, r, v = collect_rollout(
                        model, env_eval, hocbf_eval, qp_solver, constraint,
                        x0, u0, ep_key, n_steps=min(n_steps, 100))
                    total_viol += v
                    total_r += r

                avg_r = total_r / n_eval
                label = SCENARIO_LABELS[i]
                print(f"{label}: r={avg_r:.1f} v={total_viol}/{n_eval}  ", end="")
                sys.stdout.flush()
            print()
            sys.stdout.flush()

    return model, gp


if __name__ == "__main__":
    train_phase3()
