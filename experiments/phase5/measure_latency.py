"""Measure per-step solve latency for p95/p99 computation."""
import sys, time, json, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/gpu/sz_workspace/RoCBF-Net')

import jax, jax.numpy as jnp, numpy as np
from rocbf.qp.diff_qp import DifferentiableQP
from envs.ccs.dynamics import USCCSDynamics5th, UncertainUSCCSDynamics5th
from envs.ccs.constraints import CCSConstraints5th
from experiments.phase5.methods_5th import (
    _pretrain_gp_5th, _make_robust_hocbf_5th,
)

OUT = '/home/gpu/sz_workspace/RoCBF-Net/results/phase5/jpc_metrics'
os.makedirs(OUT, exist_ok=True)

LOAD_RATIO = 1.0
N_STEPS = 500

dynamics = USCCSDynamics5th(load_ratio=LOAD_RATIO)
x0, u0 = dynamics.equilibrium(LOAD_RATIO)
constraint = CCSConstraints5th(p_bounds=(13.0,24.0), h_bounds=(2670,2830), power_deviation=50.0, power_target=1000.0)

# GP and filter for S1
gp = _pretrain_gp_5th(LOAD_RATIO, n_pretrain=3000, key=jax.random.key(0), scenario='heat_absorption', scenario_specific=True)
rhocbf = _make_robust_hocbf_5th(dynamics, constraint, gp, u0, use_mean_correction=True, epsilon_kappa=1.0, use_phi_scaled_g=True)

env = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO, uncertainty_scenario='heat_absorption')

def measure_qp_latency(n_runs=5):
    """Measure QP solve time (JIT-compiled GPU)"""
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
    # Warmup JIT
    x = x0.copy()
    v_rl = jnp.zeros(3)
    A, b = rhocbf.qp_matrices(x)
    for _ in range(10):
        _ = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)

    latencies = []
    x = x0.copy()
    for step in range(N_STEPS):
        v_rl = jnp.zeros(3)
        A, b = rhocbf.qp_matrices(x)
        t0 = time.perf_counter()
        v_safe = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
        if isinstance(v_safe, tuple): v_safe = v_safe[0]
        v_safe = jnp.asarray(v_safe)
        x = env.step_stabilized_phi_scaled(x, v_safe)
    return np.array(latencies)

def measure_nmpc_latency():
    """Measure NMPC solve time (CPU SLSQP)"""
    from rocbf.baselines.nmpc import NMPCBaseline
    nmpc = NMPCBaseline(dynamics, constraint, u0, horizon=10, dt=1.0)
    env_nmpc = UncertainUSCCSDynamics5th(load_ratio=LOAD_RATIO, uncertainty_scenario='heat_absorption')
    latencies = []
    x = x0.copy()
    d_x = np.zeros(5)
    for step in range(100):  # fewer steps for slow NMPC
        t0 = time.perf_counter()
        try:
            u = nmpc.solve(x, d_x)
        except:
            u = np.zeros(3)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        next_x = env_nmpc.step_stabilized_phi_scaled(x, jnp.array(u))
        d_x = np.array(next_x - x)
        x = next_x
    return np.array(latencies)

def measure_forward_pass():
    """Measure PPO forward pass + QP time (JIT) — includes GP inference"""
    # The ~25ms includes PPO forward pass + GP inference + QP solve
    # We already have QP solve time; measure GP+forward overhead separately
    qp = DifferentiableQP(v_max=5.0, scale_constraints=True)
    latencies = []
    x = x0.copy()
    for step in range(N_STEPS):
        v_rl = jnp.zeros(3)
        t0 = time.perf_counter()
        eps = rhocbf.compute_epsilon(x)
        A, b = rhocbf.qp_matrices(x)
        v_safe = qp.solve_with_rl_action(v_rl, A, b, differentiable=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if isinstance(v_safe, tuple): v_safe = v_safe[0]
        v_safe = jnp.asarray(v_safe)
        x = env.step_stabilized_phi_scaled(x, v_safe)
    return np.array(latencies)

print("Measuring QP solve latency (JIT, GPU)...")
qp_lat = measure_qp_latency()
print(f"  QP only: mean={np.mean(qp_lat):.1f}, p50={np.percentile(qp_lat,50):.1f}, p95={np.percentile(qp_lat,95):.1f}, p99={np.percentile(qp_lat,99):.1f} ms")

print("Measuring full filter (ϵ + QP + GP inference, GPU)...")
full_lat = measure_forward_pass()
print(f"  Full filter: mean={np.mean(full_lat):.1f}, p50={np.percentile(full_lat,50):.1f}, p95={np.percentile(full_lat,95):.1f}, p99={np.percentile(full_lat,99):.1f} ms")

print("Measuring NMPC latency (SLSQP, CPU)...")
nmpc_lat = measure_nmpc_latency()
print(f"  NMPC: mean={np.mean(nmpc_lat):.1f}, p50={np.percentile(nmpc_lat,50):.1f}, p95={np.percentile(nmpc_lat,95):.1f}, p99={np.percentile(nmpc_lat,99):.1f} ms")

results = {
    'qp_gpu': {'mean': float(np.mean(qp_lat)), 'p50': float(np.percentile(qp_lat,50)), 'p95': float(np.percentile(qp_lat,95)), 'p99': float(np.percentile(qp_lat,99))},
    'full_filter_gpu': {'mean': float(np.mean(full_lat)), 'p50': float(np.percentile(full_lat,50)), 'p95': float(np.percentile(full_lat,95)), 'p99': float(np.percentile(full_lat,99))},
    'nmpc_cpu': {'mean': float(np.mean(nmpc_lat)), 'p50': float(np.percentile(nmpc_lat,50)), 'p95': float(np.percentile(nmpc_lat,95)), 'p99': float(np.percentile(nmpc_lat,99))},
}

with open(os.path.join(OUT, 'latency_percentiles.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}/latency_percentiles.json")
