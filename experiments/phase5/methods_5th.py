"""Method factory and training functions for all 8 methods on 5th-order CCS.

5th-order model: x = [r_B, p_m, h_m, N_e, τ_f] (5 states)
6 CBF constraints: pressure high/low (m=2), enthalpy high/low (m=1), power high/low (m=1)

Key differences from 3rd-order methods.py:
- n_obs=5 (was 3)
- 6 CBF constraints including power (was 4, power was rd=0)
- GP n_dims=3 (models core state residuals only; RobustHOCBF slices x[:3])
- No delay_order needed (τ_f is explicit state)
- constraint.check_all(x) without u argument (power is state-based)
- step_stabilized_phi_scaled for nonlinear Φ-scaled rollout (vs linear in Phase 4)
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
from rocbf.baselines.nmpc_5th import NMPCController5th
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.common_5th import collect_gp_data_5th as _collect_gp_core_5th

SCENARIOS = [None, "heat_absorption", "pressure_oscillation", "coupled", "nonlinear",
             "valve_degradation", "fuel_quality"]
SCENARIO_LABELS = ["Nominal", "S1:Heat", "S2:Pressure", "S3:Coupled", "S4:Nonlinear",
                   "S5:Valve", "S6:Fuel"]

NX = 5          # State dimension for 5th-order model
N_GP_DIMS = 3   # GP models residuals on core states (r_B, p_m, h_m) only


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
    x0, _ = dynamics.equilibrium(dynamics._load_ratio)
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_power),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=g_fn, relative_degree=1, k_gains=list(k_power),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa,
                     epsilon_floor=epsilon_floor, use_mean_correction=use_mean_correction),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def _pretrain_gp_5th(load_ratio, n_pretrain=500, key=None,
                      sigma_floor=1e-4, scenario=None, scenario_specific=False,
                      gp_coverage='full'):
    """Pre-train GP on 5th-order CCS scenarios.

    GP models residuals on the 3 core states (r_B, p_m, h_m) only.
    RobustHOCBF slices x[:3] when calling gp.predict(), so the GP MUST
    be trained with n_dims=3.

    Uses collect_gp_data_5th from common_5th.py which correctly handles
    5D → 3D state slicing during data collection.

    Parameters
    ----------
    scenario : str or None
        Scenario for data collection (None = nominal).
    scenario_specific : bool
        If True, train GP only on the given scenario.
        If False, train on mixed scenarios (S1-S4).
    gp_coverage : str
        'full': wide coverage (default n_pretrain)
        'sparse': near-equilibrium only (n_pretrain ≤ 200)
        'moderate': intermediate coverage (n_pretrain ≤ 500)
    """
    if key is None:
        key = jax.random.key(42)

    if gp_coverage == 'sparse':
        n_pretrain = min(n_pretrain, 200)
    elif gp_coverage == 'moderate':
        n_pretrain = min(n_pretrain, 500)

    X_all, Y_all = [], []
    if scenario_specific:
        scenarios_to_train = [scenario]
    else:
        scenarios_to_train = SCENARIOS[:4]  # S1-S4 for mixed GP

    per_scenario = n_pretrain // len(scenarios_to_train)
    for sc in scenarios_to_train:
        if sc is None:
            env = USCCSDynamics5th(dt=1.0, load_ratio=load_ratio)
        else:
            env = UncertainUSCCSDynamics5th(
                dt=1.0, load_ratio=load_ratio,
                uncertainty_scenario=sc)
        key, data_key = jax.random.split(key)
        X, Y = _collect_gp_core_5th(env, n_transitions=per_scenario,
                                      key=data_key, load_ratio=load_ratio)
        X_all.append(X)
        Y_all.append(Y)

    X_combined = jnp.concatenate(X_all, axis=0)
    Y_combined = jnp.concatenate(Y_all, axis=0)
    gp = GPResidual(n_dims=N_GP_DIMS, noise_variance=1e-4,
                     sigma_floor=sigma_floor)
    gp.fit(X_combined, Y_combined)
    return gp


# ---------- Constraint classification ----------

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
    """Rollout with 6-constraint QP safety filter on 5th-order CCS.

    State handling:
    - Actor sees full 5D state x[:NX]
    - CBF QP matrices use core 3D state x[:3] (GP + constraints on core states)
    - Dynamics step uses full 5D state x[:NX]
    """
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

        # QP safety filter (CBF uses core 3D state)
        t0 = time.perf_counter()
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:NX])
        else:
            A, b = multi_hocbf.qp_matrices(x[:NX])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        # Φ-scaled nonlinear rollout with full 5D state
        next_x = dynamics.step_stabilized_phi_scaled(x[:NX], v_safe)
        constraint_vals = constraint.check_all(next_x)

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


def _rollout_lqr_5th(dynamics, multi_hocbf, qp_solver, constraint,
                      x0, u0, key, n_steps=300, jit_qp_fn=None):
    """LQR-only rollout: v_rl = 0, QP filter does all safety work.

    This is the controller-agnostic baseline: can the safety filter
    alone guarantee forward invariance with zero-deviation input?
    """
    violations = 0
    cbf_violations = 0
    qp_times = []
    x = x0

    for t in range(n_steps):
        v_rl = jnp.zeros(3)
        t0 = time.perf_counter()
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:NX])
        else:
            A, b = multi_hocbf.qp_matrices(x[:NX])
        v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        next_x = dynamics.step_stabilized_phi_scaled(x[:NX], v_safe)
        constraint_vals = constraint.check_all(next_x)

        if _count_violations_5th(constraint_vals, protected_only=False):
            violations += 1
        if _count_violations_5th(constraint_vals, protected_only=True):
            cbf_violations += 1
        x = next_x

    return violations, cbf_violations, qp_times


# ---------- Training functions ----------

def train_ppo_5th(config, dynamics, constraint, key, gp=None):
    """Method 1: Pure PPO (no safety layer) on 5th-order CCS."""
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    return model, trainer, None


def train_ppo_lagr_5th(config, dynamics, constraint, key, gp=None):
    """Method 2: PPO-Lagrangian (soft constraint via dual descent)."""
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainerLagrangian(
        model, lr=config.get('lr', 1e-4),
        cost_limit=config.get('cost_limit', 0.0),
        lagrangian_lr=config.get('lagrangian_lr', 0.01),
        epochs=config.get('epochs', 4),
        minibatch_size=config.get('minibatch_size', 64))
    return model, trainer, None


def make_nmpc_5th(config, dynamics, constraint, key=None, gp=None):
    """Method 3: NMPC controller on 5th-order (no training needed)."""
    nmpc = NMPCController5th(
        dynamics=dynamics, constraint=constraint,
        horizon=config.get('horizon', 10),
        Q=config.get('Q'), R=config.get('R'))
    return None, None, nmpc


def train_ppo_cbf_5th(config, dynamics, constraint, key, gp=None):
    """Method 4: PPO + first-order CBF (m=1 for ALL constraints — deliberate ablation).

    On 5th-order, this incorrectly uses m=1 for pressure (should be m=2),
    but correctly uses m=1 for enthalpy and power. 6 constraints total.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    # First-order CBF: m=1 for all 6 constraints (pressure m=2 deliberately dropped)
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_power_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
        HOCBF(h_fn=constraint.h_power_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=[1.0], u0=u0),
    ]
    safety_layer = MultiConstraintHOCBF(hocbf_list)
    return model, trainer, safety_layer


def train_ppo_hocbf_5th(config, dynamics, constraint, key, gp=None):
    """Method 5: PPO + HOCBF (correct relative degrees, no GP)."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    k_p = tuple(config.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(config.get('enthalpy_k_gains', (1.0,)))
    k_n = tuple(config.get('power_k_gains', (1.0,)))
    safety_layer = _make_hocbf_5th(dynamics, constraint, u0,
                                    k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
                                    use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_ppo_gp_hocbf_5th(config, dynamics, constraint, key, gp=None):
    """Method 6: PPO + GP-corrected HOCBF (mean correction, ε=0).

    Uses RobustHOCBF with use_mean_correction=True and epsilon_kappa=0.0.
    This tests whether GP mean correction alone (without robust margin) helps.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               n_pretrain=config.get('n_pretrain', 500),
                               sigma_floor=config.get('sigma_floor', 1e-4),
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
        epsilon_kappa=0.0, k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=config.get('u_max', 100.0),
        use_mean_correction=True, epsilon_floor=0.0,
        use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_ppo_rhocbf_5th(config, dynamics, constraint, key, gp=None):
    """Method 7: PPO + Robust HOCBF (full theoretical bound, ε_κ=1.0, fixed GP).

    Uses scenario-specific GP with MC=True and compositional ε(x).
    The GP is fixed after pretraining (no online adaptation).
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               n_pretrain=config.get('n_pretrain', 500),
                               sigma_floor=config.get('sigma_floor', 1e-4),
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
        epsilon_kappa=config.get('epsilon_kappa', 1.0),
        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=config.get('u_max', 100.0),
        use_mean_correction=config.get('use_mean_correction', True),
        epsilon_floor=config.get('epsilon_floor', 0.0),
        use_phi_scaled_g=True)
    return model, trainer, safety_layer


def train_rocbf_net_5th(config, dynamics, constraint, key, gp=None):
    """Method 8: RoCBF-Net (ours) — Robust HOCBF with online GP adaptation.

    Key advantage over PPO-RHOCBF: online GP adaptation maintains an
    up-to-date uncertainty model during deployment.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp_5th(dynamics._load_ratio, key=key,
                               n_pretrain=config.get('n_pretrain', 500),
                               sigma_floor=config.get('sigma_floor', 1e-4),
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
        epsilon_kappa=config.get('epsilon_kappa', 1.0),
        k_pressure=k_p, k_enthalpy=k_h, k_power=k_n,
        u_max=config.get('u_max', 100.0),
        use_mean_correction=config.get('use_mean_correction', True),
        epsilon_floor=config.get('epsilon_floor', 0.0),
        use_phi_scaled_g=True)
    return model, trainer, safety_layer


def make_lqr_rhocbf_5th(dynamics, constraint, gp, u0,
                          epsilon_kappa=1.0, epsilon_floor=0.0,
                          k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), k_power=(1.0,),
                          u_max=100.0, use_mean_correction=True):
    """LQR-RHOCBF: LQR controller (v=0) + Robust HOCBF safety filter.

    This is the controller-agnostic baseline: purely reactive safety
    with no learned anticipatory action.
    """
    safety_layer = _make_robust_hocbf_5th(
        dynamics, constraint, gp, u0,
        epsilon_kappa=epsilon_kappa,
        k_pressure=k_pressure, k_enthalpy=k_enthalpy, k_power=k_power,
        u_max=u_max, use_mean_correction=use_mean_correction,
        epsilon_floor=epsilon_floor, use_phi_scaled_g=True)
    return safety_layer


# ---------- Method registry ----------

METHODS_5TH = {
    'ppo': train_ppo_5th,
    'ppo_lagr': train_ppo_lagr_5th,
    'nmpc': make_nmpc_5th,
    'ppo_cbf': train_ppo_cbf_5th,
    'ppo_hocbf': train_ppo_hocbf_5th,
    'ppo_gp_hocbf': train_ppo_gp_hocbf_5th,
    'ppo_rhocbf': train_ppo_rhocbf_5th,
    'rocbf_net': train_rocbf_net_5th,
}

METHOD_LABELS = {
    'ppo': 'PPO',
    'ppo_lagr': 'PPO-Lagrangian',
    'nmpc': 'NMPC',
    'ppo_cbf': 'PPO-CBF',
    'ppo_hocbf': 'PPO-HOCBF',
    'ppo_gp_hocbf': 'PPO-GP-HOCBF',
    'ppo_rhocbf': 'PPO-RHOCBF',
    'rocbf_net': 'RoCBF-Net (Ours)',
}
