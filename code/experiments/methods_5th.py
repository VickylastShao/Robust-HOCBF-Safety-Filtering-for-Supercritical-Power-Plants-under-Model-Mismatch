"""Method factory and training functions for all 8 methods on 5th-order CCS.

5th-order model: x = [r_B, p_m, h_m, N_e, τ_f] (5 states)
6 CBF constraints: pressure high/low (m=2), enthalpy high/low (m=1), power high/low (m=1)

Key differences from 3rd-order methods.py:
- n_obs=5 (was 3)
- 6 CBF constraints including power (was 4, power was rd=0)
- GP n_dims=5 (was 3)
- No delay_order needed (τ_f is explicit state)
- constraint.check_all(x) without u argument (power is state-based)
"""
import sys
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.multi_hocbf import MultiConstraintHOCBF, MultiConstraintRobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual
from rocbf.baselines.ppo_lagrangian import PPOTrainerLagrangian, compute_step_costs
from rocbf.baselines.ppo_cbf import make_first_order_cbf
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th


SCENARIOS = [None, "heat_absorption", "pressure_oscillation", "coupled", "nonlinear",
             "valve_degradation", "fuel_quality"]
SCENARIO_LABELS = ["Nominal", "S1:Heat", "S2:Pressure", "S3:Coupled", "S4:Nonlinear",
                   "S5:Valve", "S6:Fuel"]

NX = 5  # State dimension for 5th-order model


def _make_ccs_env_5th(load_ratio, scenario=None):
    """Create 5th-order dynamics and constraint for a given condition."""
    if scenario is not None:
        dynamics = UncertainUSCCSDynamics5th(
            dt=1.0, load_ratio=load_ratio,
            uncertainty_scenario=scenario)
    else:
        dynamics = USCCSDynamics5th(
            dt=1.0, load_ratio=load_ratio)
    constraint = CCSConstraints5th(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0)
    return dynamics, constraint


def _make_hocbf_5th(dynamics, constraint, u0,
                    k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
                    use_phi_scaled_g=False):
    """Create HOCBF with 6 constraints using linearized stabilized drift.

    Parameters
    ----------
    use_phi_scaled_g : bool
        If True, use g_phi_scaled (state-dependent, matches Φ-scaled rollout).
        If False, use g_linear (constant, matches linear rollout).
    """
    g_fn = dynamics.g_phi_scaled if use_phi_scaled_g else dynamics.g_linear
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure), u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure), u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy), u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy), u0=u0),
        HOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=1, k_gains=list(k_power), u0=u0),
        HOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=g_fn, relative_degree=1, k_gains=list(k_power), u0=u0),
    ]
    return MultiConstraintHOCBF(hocbf_list)


def _make_robust_hocbf_5th(dynamics, constraint, gp, u0, epsilon_kappa=1.0,
                            k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
                            u_max=100.0, use_mean_correction=False, epsilon_floor=0.0,
                            use_phi_scaled_g=False):
    """Create RobustHOCBF with 6 constraints.

    Parameters
    ----------
    use_phi_scaled_g : bool
        If True, use g_phi_scaled (state-dependent, matches Φ-scaled rollout).
        If False, use g_linear (constant, matches linear rollout).
    """
    g_fn = dynamics.g_phi_scaled if use_phi_scaled_g else dynamics.g_linear
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_power),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_power),
                     gp_residual=gp, u_max=u_max, u0=u0, epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def _collect_gp_data_5th(dynamics, n_transitions=500, key=None,
                          state_range=None, action_range=None):
    """Collect GP training data from 5th-order stabilized dynamics rollouts."""
    if key is None:
        key = jax.random.key(0)
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)

    # Default ranges for 5D state
    if state_range is None:
        max_dev = jnp.array([30.0, 5.0, 300.0, 50.0, 10.0])
        reset_noise = jnp.array([5.0, 0.5, 50.0, 10.0, 2.0])
    else:
        max_dev, reset_noise = state_range

    if action_range is None:
        v_min = jnp.array([-2.0, -5.0, -1.0])
        v_max = jnp.array([2.0, 5.0, 1.0])
    else:
        v_min, v_max = action_range

    X_list, Y_list = [], []
    x = x0
    for _ in range(n_transitions):
        key, v_key = jax.random.split(key)
        v = jnp.array([
            jax.random.uniform(v_key, (), minval=float(v_min[i]), maxval=float(v_max[i]))
            for i in range(3)
        ])
        x_next = dynamics.step_stabilized_phi_scaled(x, v)
        # Residual: difference between Φ-scaled step and Φ-scaled prediction
        # (captures only the perturbation Δf, NOT the control effectiveness mismatch)
        phi_ratio = dynamics.fluid_property(x[1]) / dynamics.fluid_property(dynamics._x0[1])
        scaling = jnp.array([1.0, phi_ratio, phi_ratio, phi_ratio, 1.0])
        B_d_phi = dynamics._B_d * scaling[:, None]
        x_pred = dynamics._x0 + dynamics._A_d @ (x[:NX] - dynamics._x0) + B_d_phi @ v
        residual = (x_next[:NX] - x_pred) / dynamics.dt
        X_list.append(x[:NX])
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next[:NX] - x0) > max_dev):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise * jax.random.normal(reset_key, (NX,))
        else:
            x = x_next
    return jnp.stack(X_list), jnp.stack(Y_list)


def _pretrain_gp_5th(load_ratio, n_pretrain=3000, key=None,
                      sigma_floor=None, scenario=None, scenario_specific=False,
                      gp_coverage='full'):
    """Pre-train GP on 5th-order CCS scenarios."""
    if key is None:
        key = jax.random.key(42)

    if gp_coverage == 'sparse':
        n_pretrain = min(n_pretrain, 200)
    elif gp_coverage == 'moderate':
        n_pretrain = min(n_pretrain, 500)

    # State ranges for different coverage levels
    if gp_coverage == 'sparse':
        state_range = (
            jnp.array([5.0, 0.5, 30.0, 10.0, 2.0]),
            jnp.array([1.0, 0.1, 10.0, 3.0, 0.5]),
        )
        action_range = (
            jnp.array([-0.3, -0.8, -0.1]),
            jnp.array([0.3, 0.8, 0.1]),
        )
    elif gp_coverage == 'moderate':
        state_range = (
            jnp.array([15.0, 2.0, 150.0, 30.0, 5.0]),
            jnp.array([3.0, 0.3, 30.0, 5.0, 1.0]),
        )
        action_range = (
            jnp.array([-1.0, -2.5, -0.5]),
            jnp.array([1.0, 2.5, 0.5]),
        )
    else:
        state_range = None
        action_range = None

    X_all, Y_all = [], []
    if scenario_specific:
        scenarios_to_train = [scenario]
    else:
        # Use first 4 scenarios for mixed GP (same as 3rd-order)
        scenarios_to_train = SCENARIOS[:4]
    per_scenario = n_pretrain // len(scenarios_to_train)
    for sc in scenarios_to_train:
        env = UncertainUSCCSDynamics5th(
            dt=1.0, load_ratio=load_ratio,
            uncertainty_scenario=sc)
        key, data_key = jax.random.split(key)
        X, Y = _collect_gp_data_5th(env, n_transitions=per_scenario, key=data_key,
                                     state_range=state_range, action_range=action_range)
        X_all.append(X)
        Y_all.append(Y)
    X_combined = jnp.concatenate(X_all, axis=0)
    Y_combined = jnp.concatenate(Y_all, axis=0)
    gp = GPResidual(n_dims=NX, noise_variance=1e-4, sigma_floor=sigma_floor)
    gp.fit(X_combined, Y_combined)
    return gp


# ---------- Constraint classification ----------
# In 5th-order model, ALL constraints are CBF-enforceable (including power)
CBF_PROTECTED_5TH = {
    'pressure_high', 'pressure_low',
    'enthalpy_high', 'enthalpy_low',
    'power_high', 'power_low',
}


def _count_violations_5th(constraint_vals, protected_only=False):
    """Count steps where any constraint is violated (5th-order)."""
    if protected_only:
        return any(v < 0 for k, v in constraint_vals.items() if k in CBF_PROTECTED_5TH)
    return any(v < 0 for v in constraint_vals.values())


# ---------- Rollout functions ----------

def _rollout_with_qp_5th(model, dynamics, multi_hocbf, qp_solver, constraint,
                          x0, u0, key, n_steps=300, jit_qp_fn=None):
    """Rollout with 6-constraint QP safety filter on 5th-order CCS."""
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': [],
               'constraint_vals': []}

    x = x0
    total_reward = 0.0
    violations = 0
    cbf_violations = 0
    qp_times = []

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:NX], action_key)

        # QP safety filter
        t0 = time.perf_counter()
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:NX])
        else:
            A, b = multi_hocbf.qp_matrices(x[:NX])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        # Step with Φ-scaled stabilized dynamics (nonlinear rollout)
        next_x = dynamics.step_stabilized_phi_scaled(x[:NX], v_safe)
        # 5th-order: check_all doesn't need u (power is state-based)
        constraint_vals = constraint.check_all(next_x)

        # Reward
        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )

        rollout['obs'].append(x[:NX])
        rollout['actions'].append(v_safe)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(0.0))
        rollout['constraint_vals'].append(constraint_vals)

        if _count_violations_5th(constraint_vals, protected_only=False):
            violations += 1
        if _count_violations_5th(constraint_vals, protected_only=True):
            cbf_violations += 1
        total_reward += float(reward)
        x = next_x

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations, cbf_violations, qp_times


def _rollout_no_qp_5th(model, dynamics, constraint, x0, u0, key, n_steps=300):
    """Rollout WITHOUT safety filter on 5th-order CCS."""
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': [],
               'constraint_vals': []}

    x = x0
    total_reward = 0.0
    violations = 0
    cbf_violations = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:NX], action_key)

        next_x = dynamics.step_stabilized_phi_scaled(x[:NX], v_rl)
        constraint_vals = constraint.check_all(next_x)

        y = dynamics.output(next_x)
        y0 = dynamics.output(x0)
        reward = (
            -1.0 * (y[0] - y0[0]) ** 2
            - 0.001 * (y[1] - y0[1]) ** 2
            - 0.01 * (y[2] - y0[2]) ** 2
            - 0.0001 * jnp.sum(v_rl ** 2)
        )

        rollout['obs'].append(x[:NX])
        rollout['actions'].append(v_rl)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(0.0))
        rollout['constraint_vals'].append(constraint_vals)

        if _count_violations_5th(constraint_vals, protected_only=False):
            violations += 1
        if _count_violations_5th(constraint_vals, protected_only=True):
            cbf_violations += 1
        total_reward += float(reward)
        x = next_x

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations, cbf_violations, []


# ---------- Training functions ----------

def train_ppo_5th(config, dynamics, constraint, key, gp=None):
    """Pure PPO on 5th-order CCS."""
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    return model, trainer, None


def train_ppo_hocbf_5th(config, dynamics, constraint, key, gp=None):
    """PPO + HOCBF on 5th-order CCS (6 constraints, no GP).

    Uses Φ-scaled g function to match nonlinear rollout dynamics.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    safety_layer = _make_hocbf_5th(dynamics, constraint, u0, use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_ppo_gp_hocbf_5th(config, dynamics, constraint, key, gp=None):
    """PPO + GP-corrected HOCBF on 5th-order CCS (mean correction, no epsilon).

    Uses Φ-scaled g function to match nonlinear rollout dynamics.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               scenario=config.get('scenario', None),
                               scenario_specific=config.get('scenario_specific_gp', True))
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    safety_layer = _make_robust_hocbf_5th(dynamics, constraint, gp, u0,
                                           epsilon_kappa=0.0,
                                           use_mean_correction=True,
                                           use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_ppo_rhocbf_5th(config, dynamics, constraint, key, gp=None):
    """PPO + Robust HOCBF on 5th-order CCS (full bound, fixed GP).

    Uses Φ-scaled g function to match nonlinear rollout dynamics.
    Default epsilon_kappa=0.5 for optimal balance of safety and stability
    on Φ-scaled nonlinear dynamics (κ=1.0 causes oscillation on S3:Coupled).
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               scenario=config.get('scenario', None),
                               scenario_specific=config.get('scenario_specific_gp', True))
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    k_p = tuple(config.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(config.get('enthalpy_k_gains', (1.0,)))
    k_n = tuple(config.get('power_k_gains', (1.0,)))
    safety_layer = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=config.get('epsilon_kappa', 0.5),
        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=config.get('u_max', 100.0),
        use_mean_correction=config.get('use_mean_correction', True),
        epsilon_floor=config.get('epsilon_floor', 0.0),
        use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_rocbf_net_5th(config, dynamics, constraint, key, gp=None):
    """RoCBF-Net on 5th-order CCS (online GP adaptation).

    Uses Φ-scaled g function to match nonlinear rollout dynamics.
    Default epsilon_kappa=0.5 for optimal balance of safety and stability
    on Φ-scaled nonlinear dynamics.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               scenario=config.get('scenario', None),
                               scenario_specific=config.get('scenario_specific_gp', True))
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    k_p = tuple(config.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(config.get('enthalpy_k_gains', (1.0,)))
    k_n = tuple(config.get('power_k_gains', (1.0,)))
    safety_layer = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=config.get('epsilon_kappa', 0.5),
        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=config.get('u_max', 100.0),
        use_mean_correction=config.get('use_mean_correction', True),
        epsilon_floor=config.get('epsilon_floor', 0.0),
        use_phi_scaled_g=True)
    return model, trainer, safety_layer


# ---------- Method registry ----------

METHODS_5TH = {
    'ppo': train_ppo_5th,
    'ppo_hocbf': train_ppo_hocbf_5th,
    'ppo_gp_hocbf': train_ppo_gp_hocbf_5th,
    'ppo_rhocbf': train_ppo_rhocbf_5th,
    'rocbf_net': train_rocbf_net_5th,
}

METHOD_LABELS = {
    'ppo': 'PPO',
    'ppo_hocbf': 'PPO-HOCBF',
    'ppo_gp_hocbf': 'PPO-GP-HOCBF',
    'ppo_rhocbf': 'PPO-RHOCBF',
    'rocbf_net': 'RoCBF-Net (Ours)',
}
