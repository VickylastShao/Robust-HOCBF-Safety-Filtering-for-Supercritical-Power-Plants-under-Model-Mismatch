"""Test candidate perturbation magnitudes with PPO + RHOCBF on 5th-order CCS."""
import sys, warnings, time, math
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np, flax.nnx as nnx
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    NX, _make_robust_hocbf_5th, _pretrain_gp_5th,
    _rollout_no_qp_5th, _count_violations_5th,
)

LOAD_RATIO = 0.75
N_EVAL = 300

d = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = d.equilibrium(LOAD_RATIO)
c = CCSConstraints5th(p_bounds=(13.0, 24.0), h_bounds=(2670, 2830),
                       power_deviation=50.0, power_target=LOAD_RATIO * 1000.0)

# Train PPO
print("Training PPO...")
model = ActorCritic(n_obs=NX, n_act=3, hidden_dim=128, rngs=nnx.Rngs(0))
trainer = PPOTrainer(model, lr=1e-4, epochs=4, minibatch_size=64)
key = jax.random.key(0)
for ep in range(30):
    key, rk = jax.random.split(key)
    rollout, ep_r, _, _, _ = _rollout_no_qp_5th(model, d, c, x0, u0, rk, n_steps=100)
    if rollout['obs'].shape[0] > 1:
        adv, ret = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'])
        batch = {'obs': rollout['obs'], 'actions': rollout['actions'],
                 'old_log_probs': rollout['log_probs'], 'advantages': adv, 'returns': ret}
        for _ in range(trainer.epochs):
            trainer.train_step(batch)
print("PPO trained.")


def eval_with_perturbation(delta_f_fn, safety_layer=None, n_eval=N_EVAL):
    """Evaluate with a custom perturbation function delta_f(t, x, x0) -> array(5)."""
    qp = DifferentiableQP(v_max=10.0)
    x = x0[:NX].copy()
    cbf_viols = 0
    qp_interventions = 0
    for t in range(n_eval):
        v_rl, _, _ = model.get_action(x, jax.random.key(t * 1000 + 42))

        if safety_layer is not None:
            A, b = safety_layer.qp_matrices(x)
            v_safe, _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
            v_safe = jnp.clip(v_safe, -10.0, 10.0)
            if jnp.any(jnp.abs(v_safe - v_rl) > 1e-3):
                qp_interventions += 1
        else:
            v_safe = v_rl

        pert = delta_f_fn(t, x, x0)
        dx = x - x0
        dx_next = d._A_d @ dx + d._B_d @ v_safe + 1.0 * pert
        x = x0 + dx_next
        x = jnp.array([jnp.clip(x[i], d.x_bounds[i][0], d.x_bounds[i][1]) for i in range(5)])
        cv = c.check_all(x)
        if _count_violations_5th(cv, protected_only=True):
            cbf_viols += 1
    return cbf_viols / n_eval * 100, qp_interventions / n_eval * 100 if safety_layer else 0.0


# === S6: Fuel quality (tau_f perturbation) ===
print("\n=== S6: Fuel quality (Δτ_f) ===")
for dtau in [-1.0, -1.5, -2.0]:
    def delta_f(t, x, x0, dtau=dtau):
        return jnp.array([0.0, 0.0, 0.0, 0.0, dtau])

    ppo_pct, _ = eval_with_perturbation(delta_f, safety_layer=None)

    # RHOCBF with scenario-specific GP
    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='fuel_quality', scenario_specific=True)
    safety = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
    rhocbf_pct, qp_pct = eval_with_perturbation(delta_f, safety_layer=safety)

    # GP-HOCBF (mean only, no eps)
    safety_noeps = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=0.0, use_mean_correction=True)
    gp_hocbf_pct, _ = eval_with_perturbation(delta_f, safety_layer=safety_noeps)

    print(f"  Δτ_f={dtau}: PPO={ppo_pct:.1f}%  GP-HOCBF={gp_hocbf_pct:.1f}%  RHOCBF={rhocbf_pct:.1f}% (QP={qp_pct:.1f}%)")


# === S5: Valve degradation (N_e perturbation) ===
print("\n=== S5: Valve degradation (ΔN_e) ===")
for dn in [-50, -80]:
    def delta_f(t, x, x0, dn=dn):
        return jnp.array([0.0, 0.0, 0.0, dn, 0.0])

    ppo_pct, _ = eval_with_perturbation(delta_f, safety_layer=None)

    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='valve_degradation', scenario_specific=True)
    safety = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
    rhocbf_pct, qp_pct = eval_with_perturbation(delta_f, safety_layer=safety)

    safety_noeps = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=0.0, use_mean_correction=True)
    gp_hocbf_pct, _ = eval_with_perturbation(delta_f, safety_layer=safety_noeps)

    print(f"  ΔN_e={dn}: PPO={ppo_pct:.1f}%  GP-HOCBF={gp_hocbf_pct:.1f}%  RHOCBF={rhocbf_pct:.1f}% (QP={qp_pct:.1f}%)")


# === S2: Pressure oscillation ===
print("\n=== S2: Pressure oscillation (amp) ===")
for amp in [5, 8, 10]:
    def delta_f(t, x, x0, amp=amp):
        dp = amp * math.sin(2 * math.pi * 0.005 * t)
        return jnp.array([0.0, dp, 0.0, 0.0, 0.0])

    ppo_pct, _ = eval_with_perturbation(delta_f, safety_layer=None)

    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='pressure_oscillation', scenario_specific=True)
    safety = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
    rhocbf_pct, qp_pct = eval_with_perturbation(delta_f, safety_layer=safety)

    print(f"  amp={amp}: PPO={ppo_pct:.1f}%  RHOCBF={rhocbf_pct:.1f}% (QP={qp_pct:.1f}%)")


# === S3: Coupled ===
print("\n=== S3: Coupled (scale) ===")
for scale in [3, 4, 5]:
    def delta_f(t, x, x0, scale=scale):
        dp = scale * (0.15 * (x[1] - x0[1]) + 0.3)
        dh = scale * (-0.1 * (x[2] - x0[2]) - 5.0)
        return jnp.array([0.0, dp, dh, 0.0, 0.0])

    ppo_pct, _ = eval_with_perturbation(delta_f, safety_layer=None)

    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='coupled', scenario_specific=True)
    safety = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
    rhocbf_pct, qp_pct = eval_with_perturbation(delta_f, safety_layer=safety)

    print(f"  scale={scale}: PPO={ppo_pct:.1f}%  RHOCBF={rhocbf_pct:.1f}% (QP={qp_pct:.1f}%)")


# === S4: Nonlinear ===
print("\n=== S4: Nonlinear (scale) ===")
for scale in [3, 5, 8]:
    def delta_f(t, x, x0, scale=scale):
        dp = scale * (0.01 * (x[1] - x0[1]) ** 2 + 0.5)
        dh = scale * (-5.0)
        return jnp.array([0.0, dp, dh, 0.0, 0.0])

    ppo_pct, _ = eval_with_perturbation(delta_f, safety_layer=None)

    gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(42),
                           scenario='nonlinear', scenario_specific=True)
    safety = _make_robust_hocbf_5th(d, c, gp, u0, epsilon_kappa=1.0, use_mean_correction=True)
    rhocbf_pct, qp_pct = eval_with_perturbation(delta_f, safety_layer=safety)

    print(f"  scale={scale}: PPO={ppo_pct:.1f}%  RHOCBF={rhocbf_pct:.1f}% (QP={qp_pct:.1f}%)")
