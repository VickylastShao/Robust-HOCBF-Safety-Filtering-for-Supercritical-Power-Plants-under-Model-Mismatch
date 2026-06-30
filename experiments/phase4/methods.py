"""Method factory and training functions for all 8 methods.

Methods:
1. ppo         — Pure PPO (no safety layer)
2. ppo_lagr    — PPO-Lagrangian (soft constraint via dual descent)
3. nmpc        — Nonlinear MPC (scipy SLSQP)
4. ppo_cbf     — PPO + first-order CBF (m=1 for all constraints)
5. ppo_hocbf   — PPO + HOCBF (correct relative degrees, no GP)
6. ppo_gp_hocbf — PPO + GP-corrected HOCBF (mean correction, no epsilon)
7. ppo_rhocbf  — PPO + Robust HOCBF (full theoretical bound, epsilon_kappa=1.0)
8. rocbf_net   — PPO + Robust HOCBF with practical kappa (ours)
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
from envs.ccs.dynamics import USCCSDynamics, UncertainUSCCSDynamics
from envs.ccs.constraints import CCSConstraints
from envs.ccs.agc_schedule import AGCSchedule


SCENARIOS = [None, "heat_absorption", "pressure_oscillation", "coupled", "nonlinear"]
SCENARIO_LABELS = ["Nominal", "S1:Heat", "S2:Pressure", "S3:Coupled", "S4:Nonlinear"]


def _make_ccs_env(load_ratio, delay_order, scenario=None):
    """Create dynamics and constraint for a given condition."""
    if scenario is not None:
        dynamics = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=scenario)
    else:
        dynamics = USCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio)
    constraint = CCSConstraints(
        p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
        power_deviation=50.0, power_target=load_ratio * 1000.0,
        dynamics=dynamics)
    return dynamics, constraint


def _make_hocbf(dynamics, constraint, u0, k_pressure=(0.5, 0.5), k_enthalpy=(1.0,)):
    """Create HOCBF with correct relative degrees using linearized stabilized drift.

    Uses f_linear_stabilized and g_linear which match the actual step_stabilized
    dynamics, ensuring QP matrices are well-conditioned.
    """
    hocbf_list = [
        HOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure), u0=u0),
        HOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure), u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy), u0=u0),
        HOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
              g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy), u0=u0),
    ]
    return MultiConstraintHOCBF(hocbf_list)


def _make_robust_hocbf(dynamics, constraint, gp, u0, epsilon_kappa=1.0,
                        k_pressure=(0.5, 0.5), k_enthalpy=(1.0,), u_max=100.0,
                        use_mean_correction=False, epsilon_floor=0.0):
    """Create RobustHOCBF with given epsilon_kappa using linearized stabilized drift.

    Parameters
    ----------
    use_mean_correction : bool
        If True, use f_hat = f0 + mu_GP in psi chain (Method 6: GP-HOCBF).
        If False (default), use nominal f0 in psi chain, GP sigma only for epsilon.
        For CCS, use_mean_correction=False is recommended because the GP mean
        from mixed-scenario training makes the CBF constraint infeasible at equilibrium.
    epsilon_floor : float
        Minimum robustness margin per constraint. Prevents ε from dropping
        below this value during online GP adaptation. Accounts for unmodeled
        dynamics not captured by the GP.
    """
    # Compute operator norm from linearized dynamics at equilibrium
    x0, _ = dynamics.equilibrium(dynamics._load_ratio)
    hocbf_list = [
        RobustHOCBF(h_fn=constraint.h_pressure_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa, epsilon_floor=epsilon_floor,
                     use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_pressure_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=2, k_gains=list(k_pressure),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa, epsilon_floor=epsilon_floor,
                     use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_high, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa, epsilon_floor=epsilon_floor,
                     use_mean_correction=use_mean_correction),
        RobustHOCBF(h_fn=constraint.h_enthalpy_low, f_fn=dynamics.f_linear_stabilized,
                     g_fn=dynamics.g_linear, relative_degree=1, k_gains=list(k_enthalpy),
                     gp_residual=gp, u_max=u_max, u0=u0, x0=x0,
                     epsilon_kappa=epsilon_kappa, epsilon_floor=epsilon_floor,
                     use_mean_correction=use_mean_correction),
    ]
    return MultiConstraintRobustHOCBF(hocbf_list)


def _collect_gp_data(dynamics, n_transitions=500, key=None,
                     state_range=None, action_range=None):
    """Collect GP training data from stabilized dynamics rollouts.

    Parameters
    ----------
    state_range : tuple or None
        (max_deviation, reset_noise) as (array(3), array(3)).
        max_deviation: reset if |x-x0| exceeds this. Default [30,5,300].
        reset_noise: noise scale when resetting. Default [5,0.5,50].
    action_range : tuple or None
        (v_min, v_max) as (array(3), array(3)) for deviation control v.
        Default v in [-2,2] x [-5,5] x [-1,1].
    """
    if key is None:
        key = jax.random.key(0)
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)

    # Default ranges: wide coverage of state space
    if state_range is None:
        max_dev = jnp.array([30.0, 5.0, 300.0])
        reset_noise = jnp.array([5.0, 0.5, 50.0])
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
        u = dynamics.compute_total_control(x, v)
        x_next = dynamics.step_stabilized(x, v)
        # Residual: difference between actual step and linearized prediction
        # For the stabilized system, the residual captures nonlinear effects + uncertainty
        x_pred = dynamics._x0 + dynamics._A_d @ (x[:3] - dynamics._x0) + dynamics._B_d @ v
        residual = (x_next[:3] - x_pred) / dynamics.dt
        X_list.append(x[:3])
        Y_list.append(residual)
        if jnp.any(jnp.abs(x_next[:3] - x0) > max_dev):
            key, reset_key = jax.random.split(key)
            x = x0 + reset_noise * jax.random.normal(reset_key, (3,))
        else:
            x = x_next
    return jnp.stack(X_list), jnp.stack(Y_list)


def _pretrain_gp(load_ratio, delay_order, n_pretrain=3000, key=None,
                  sigma_floor=None, scenario=None, scenario_specific=False,
                  gp_coverage='full'):
    """Pre-train GP on scenarios.

    Parameters
    ----------
    scenario : str or None
        Scenario for data collection.
    scenario_specific : bool
        If True, train GP only on the given scenario (scenario-specific GP).
        If False and scenario is None, train on all scenarios (mixed GP).
    gp_coverage : str
        'full': standard wide-coverage GP (default)
        'sparse': few data points near equilibrium only (n_pretrain=200)
                  Creates high σ_GP variation across state space.
        'moderate': intermediate coverage (n_pretrain=500)
    """
    if key is None:
        key = jax.random.key(42)

    # Override n_pretrain for sparse/moderate coverage
    if gp_coverage == 'sparse':
        n_pretrain = min(n_pretrain, 200)
    elif gp_coverage == 'moderate':
        n_pretrain = min(n_pretrain, 500)

    # Determine data range based on coverage
    if gp_coverage == 'sparse':
        # Narrow state range: keep data near equilibrium
        state_range = (
            jnp.array([5.0, 0.5, 30.0]),    # max deviation from x0
            jnp.array([1.0, 0.1, 10.0]),     # reset noise
        )
        action_range = (
            jnp.array([-0.3, -0.8, -0.1]),   # small actions
            jnp.array([0.3, 0.8, 0.1]),
        )
    elif gp_coverage == 'moderate':
        state_range = (
            jnp.array([15.0, 2.0, 150.0]),   # moderate state coverage
            jnp.array([3.0, 0.3, 30.0]),     # moderate reset noise
        )
        action_range = (
            jnp.array([-1.0, -2.5, -0.5]),   # moderate actions
            jnp.array([1.0, 2.5, 0.5]),
        )
    else:
        state_range = None  # use defaults (wide coverage)
        action_range = None

    X_all, Y_all = [], []
    if scenario_specific:
        scenarios_to_train = [scenario]  # [None] for nominal
    else:
        scenarios_to_train = SCENARIOS
    per_scenario = n_pretrain // len(scenarios_to_train)
    for sc in scenarios_to_train:
        env = UncertainUSCCSDynamics(
            delay_order=delay_order, load_ratio=load_ratio,
            uncertainty_scenario=sc)
        key, data_key = jax.random.split(key)
        X, Y = _collect_gp_data(env, n_transitions=per_scenario, key=data_key,
                                state_range=state_range, action_range=action_range)
        X_all.append(X)
        Y_all.append(Y)
    X_combined = jnp.concatenate(X_all, axis=0)
    Y_combined = jnp.concatenate(Y_all, axis=0)
    gp = GPResidual(n_dims=3, noise_variance=1e-4, sigma_floor=sigma_floor)
    gp.fit(X_combined, Y_combined)
    return gp


# ---------- Generic rollout with QP safety filter ----------

# Constraints protected by the HOCBF (relative degree > 0).
# Power constraints (relative degree 0) cannot be enforced via CBF.
CBF_PROTECTED = {'pressure_high', 'pressure_low', 'enthalpy_high', 'enthalpy_low'}


def _count_violations(constraint_vals, protected_only=False):
    """Count steps where any constraint is violated.

    Parameters
    ----------
    constraint_vals : dict
        Output of CCSConstraints.check_all().
    protected_only : bool
        If True, only count CBF-protected constraints (pressure + enthalpy).
        If False, count all constraints including power.
    """
    if protected_only:
        return any(v < 0 for k, v in constraint_vals.items() if k in CBF_PROTECTED)
    return any(v < 0 for v in constraint_vals.values())


def _rollout_with_qp(model, dynamics, multi_hocbf, qp_solver, constraint,
                     x0, u0, key, n_steps=300, agc_schedule=None,
                     use_scipy=False, jit_qp_fn=None):
    """Collect rollout data with multi-constraint QP safety filter.

    Uses deviation-form control: RL outputs v, QP filters v,
    total control u = u0 + K@(x0-x) + v_safe.
    Uses step_stabilized for numerically stable integration.

    Parameters
    ----------
    use_scipy : bool
        If True, use scipy SLSQP (robust but slow).
        If False, use qpax (fast, may produce NaN → fallback to v=0).
        Training uses qpax for speed; evaluation uses scipy for accuracy.
    jit_qp_fn : callable, optional
        JIT-compiled version of multi_hocbf.qp_matrices for speed.
        If None, uses multi_hocbf.qp_matrices directly (slow ~466ms/call).
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
        v_rl, log_prob, value = model.get_action(x[:3], action_key)

        # QP safety filter on deviation control v
        t0 = time.perf_counter()
        if jit_qp_fn is not None:
            A, b = jit_qp_fn(x[:3])
        else:
            A, b = multi_hocbf.qp_matrices(x[:3])
        if use_scipy:
            v_safe, _ = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=False)
        else:
            v_safe = qp_solver.solve_with_rl_action(v_rl, A, b, differentiable=True)
        # Safety clip: QP solver has v_max bounds but clip as defense-in-depth
        v_max = qp_solver.v_max if qp_solver.v_max else 10.0
        v_safe = jnp.clip(v_safe, -v_max, v_max)
        qp_times.append((time.perf_counter() - t0) * 1000)

        # Step with stabilized dynamics
        next_x = dynamics.step_stabilized(x[:3], v_safe)
        u_total = dynamics.compute_total_control(x[:3], v_safe)
        constraint_vals = constraint.check_all(next_x, u_total)

        # Reward: pure tracking + control cost (no violation penalty)
        if agc_schedule is not None:
            target_load = agc_schedule.get_reference(float(t))
            _, u_target = agc_schedule.get_equilibrium(target_load, dynamics)
            y = dynamics.output(next_x, u_total)
            x_ref, _ = dynamics.equilibrium(target_load / 1000.0)
            y_ref = dynamics.output(x_ref, u_target)
        else:
            y = dynamics.output(next_x, u_total)
            y0 = dynamics.output(x0, u0)
            y_ref = y0

        reward = (
            -1.0 * (y[0] - y_ref[0]) ** 2
            - 0.001 * (y[1] - y_ref[1]) ** 2
            - 0.01 * (y[2] - y_ref[2]) ** 2
            - 0.0001 * jnp.sum(v_safe ** 2)
        )

        rollout['obs'].append(x[:3])
        rollout['actions'].append(v_safe)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(0.0))
        rollout['constraint_vals'].append(constraint_vals)

        if _count_violations(constraint_vals, protected_only=False):
            violations += 1
        if _count_violations(constraint_vals, protected_only=True):
            cbf_violations += 1
        total_reward += float(reward)
        x = next_x

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations, cbf_violations, qp_times


# ---------- Generic rollout without QP (pure PPO / PPO-Lagrangian) ----------

def _rollout_no_qp(model, dynamics, constraint, x0, u0, key, n_steps=300,
                   agc_schedule=None):
    """Collect rollout data WITHOUT safety filter (pure RL).

    Uses deviation-form control: RL outputs v,
    total control u = u0 + K@(x0-x) + v.
    Uses step_stabilized for numerically stable integration.
    """
    rollout = {'obs': [], 'actions': [], 'rewards': [],
               'log_probs': [], 'values': [], 'dones': [],
               'constraint_vals': []}

    x = x0
    total_reward = 0.0
    violations = 0
    cbf_violations = 0

    for t in range(n_steps):
        key, action_key = jax.random.split(key)
        v_rl, log_prob, value = model.get_action(x[:3], action_key)

        # Step with stabilized dynamics (no QP filter)
        next_x = dynamics.step_stabilized(x[:3], v_rl)
        u_total = dynamics.compute_total_control(x[:3], v_rl)
        constraint_vals = constraint.check_all(next_x, u_total)

        if agc_schedule is not None:
            target_load = agc_schedule.get_reference(float(t))
            x_ref, u_target = dynamics.equilibrium(target_load / 1000.0)
            y_ref = dynamics.output(x_ref, u_target)
        else:
            y0 = dynamics.output(x0, u0)
            y_ref = y0

        y = dynamics.output(next_x, u_total)
        reward = (
            -1.0 * (y[0] - y_ref[0]) ** 2
            - 0.001 * (y[1] - y_ref[1]) ** 2
            - 0.01 * (y[2] - y_ref[2]) ** 2
            - 0.0001 * jnp.sum(v_rl ** 2)
        )

        rollout['obs'].append(x[:3])
        rollout['actions'].append(v_rl)
        rollout['rewards'].append(reward)
        rollout['log_probs'].append(log_prob)
        rollout['values'].append(value)
        rollout['dones'].append(jnp.float32(0.0))
        rollout['constraint_vals'].append(constraint_vals)

        if _count_violations(constraint_vals, protected_only=False):
            violations += 1
        if _count_violations(constraint_vals, protected_only=True):
            cbf_violations += 1
        total_reward += float(reward)
        x = next_x

    for k in ['obs', 'actions', 'rewards', 'log_probs', 'values', 'dones']:
        rollout[k] = jnp.stack(rollout[k])

    return rollout, total_reward, violations, cbf_violations, []


# ---------- Training functions for each method ----------

def train_ppo(config, dynamics, constraint, key, gp=None):
    """Method 1: Pure PPO (no safety layer)."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    return model, trainer, None  # no safety layer


def train_ppo_lagrangian(config, dynamics, constraint, key, gp=None):
    """Method 2: PPO-Lagrangian (soft constraint)."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainerLagrangian(
        model, lr=config.get('lr', 1e-4),
        cost_limit=config.get('cost_limit', 0.0),
        lagrangian_lr=config.get('lagrangian_lr', 0.01),
        epochs=config.get('epochs', 4),
        minibatch_size=config.get('minibatch_size', 64))
    return model, trainer, None


def make_nmpc(config, dynamics, constraint, key=None, gp=None):
    """Method 3: NMPC controller (no training needed)."""
    from rocbf.baselines.nmpc import NMPCController
    nmpc = NMPCController(
        dynamics=dynamics, constraint=constraint,
        horizon=config.get('horizon', 20),
        Q=None, R=None)
    return None, None, nmpc  # no model/trainer, just nmpc controller


def train_ppo_cbf(config, dynamics, constraint, key, gp=None):
    """Method 4: PPO + first-order CBF (m=1 for all constraints)."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    safety_layer = make_first_order_cbf(constraint, dynamics, u0)
    return model, trainer, safety_layer


def train_ppo_hocbf(config, dynamics, constraint, key, gp=None):
    """Method 5: PPO + HOCBF (correct relative degrees, no GP)."""
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    safety_layer = _make_hocbf(dynamics, constraint, u0)
    return model, trainer, safety_layer


def train_ppo_gp_hocbf(config, dynamics, constraint, key, gp=None):
    """Method 6: PPO + GP-corrected HOCBF (mean correction, no epsilon).

    Uses RobustHOCBF with use_mean_correction=True and epsilon_kappa=0.0.
    This tests whether GP mean correction alone (without robust margin) helps.
    Note: for CCS with mixed-scenario GP, this can make the CBF constraint
    infeasible at equilibrium, which is a key ablation finding.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp(dynamics._load_ratio, dynamics.delay_order, key=key)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0, epsilon_kappa=0.0,
                                       use_mean_correction=True)
    return model, trainer, safety_layer


def train_ppo_rhocbf(config, dynamics, constraint, key, gp=None):
    """Method 7: PPO + Robust HOCBF (full theoretical bound, epsilon_kappa=1.0).

    Uses scenario-specific GP with MC=True to account for perturbation mean
    in the drift model, and epsilon(x) to cover residual stochastic uncertainty.
    The GP is fixed after pretraining (no online adaptation).
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp(dynamics._load_ratio, dynamics.delay_order, key=key)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    kappa = config.get('epsilon_kappa', 1.0)
    use_mc = config.get('use_mean_correction', True)
    epsilon_floor = config.get('epsilon_floor', 0.0)
    k_p = tuple(config.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(config.get('enthalpy_k_gains', (1.0,)))
    u_max = config.get('u_max', 100.0)
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                       epsilon_kappa=kappa,
                                       k_pressure=k_p, k_enthalpy=k_h,
                                       u_max=u_max,
                                       use_mean_correction=use_mc,
                                       epsilon_floor=epsilon_floor)
    return model, trainer, safety_layer


def train_rocbf_net(config, dynamics, constraint, key, gp=None):
    """Method 8: RoCBF-Net (ours) — online GP adaptation for safe RL.

    Uses scenario-specific GP with MC=True to account for perturbation mean
    in the drift model, and online GP updates during deployment to track
    evolving uncertainty. The mean-corrected drift ensures the QP enforces
    corrective actions against systematic perturbations, while epsilon(x)
    covers residual stochastic uncertainty.

    Key advantage over PPO-RHOCBF: online GP adaptation maintains an
    up-to-date uncertainty model during deployment, which is critical for
    time-varying and evolving perturbation scenarios.
    """
    x0, u0 = dynamics.equilibrium(dynamics._load_ratio)
    if gp is None:
        gp = _pretrain_gp(dynamics._load_ratio, dynamics.delay_order, key=key)
    model = ActorCritic(n_obs=3, n_act=3, hidden_dim=config.get('hidden_dim', 128),
                        rngs=nnx.Rngs(0))
    trainer = PPOTrainer(model, lr=config.get('lr', 1e-4),
                         epochs=config.get('epochs', 4),
                         minibatch_size=config.get('minibatch_size', 64))
    epsilon_kappa = config.get('epsilon_kappa', 1.0)
    epsilon_floor = config.get('epsilon_floor', 0.0)
    use_mean_correction = config.get('use_mean_correction', True)
    k_p = tuple(config.get('pressure_k_gains', (0.5, 0.5)))
    k_h = tuple(config.get('enthalpy_k_gains', (1.0,)))
    u_max = config.get('u_max', 100.0)
    safety_layer = _make_robust_hocbf(dynamics, constraint, gp, u0,
                                       epsilon_kappa=epsilon_kappa,
                                       k_pressure=k_p, k_enthalpy=k_h,
                                       u_max=u_max,
                                       use_mean_correction=use_mean_correction,
                                       epsilon_floor=epsilon_floor)
    return model, trainer, safety_layer


# ---------- Method registry ----------

METHODS = {
    'ppo': train_ppo,
    'ppo_lagr': train_ppo_lagrangian,
    'nmpc': make_nmpc,
    'ppo_cbf': train_ppo_cbf,
    'ppo_hocbf': train_ppo_hocbf,
    'ppo_gp_hocbf': train_ppo_gp_hocbf,
    'ppo_rhocbf': train_ppo_rhocbf,
    'rocbf_net': train_rocbf_net,
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
