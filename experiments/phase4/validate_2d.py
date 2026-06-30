"""2D Double Integrator Validation — Phase Portrait.

Visual proof of RoCBF-Net's theoretical guarantees on the simple
double integrator:
- Nominal safe set C₀ = {x : h(x) ≥ 0}
- Robust safe set C_ε = {x : h(x) ≥ ε(x)} (shrunk by robust margin)
- True safe set boundary under uncertainty

Plots:
1. Phase portrait (position vs velocity) with safe boundaries
2. Epsilon vs state plot
3. Violation comparison bar chart
"""
import jax
import jax.numpy as jnp
import numpy as np

from envs.safe_navigation.dynamics import DoubleIntegratorDynamics, UncertainDoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut
from rocbf.rl.ppo import ActorCritic, PPOTrainer, compute_gae
from rocbf.cbf.hocbf import HOCBF
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual


def setup_2d_system(dt=0.01, u_max=1.0):
    """Create 2D double integrator with circular keep-out constraint."""
    dynamics = DoubleIntegratorDynamics(dt=dt, u_max=u_max)
    constraint = CircularKeepOut(center=jnp.array([1.0, 0.0]), radius=0.5)
    return dynamics, constraint


def collect_gp_data_2d(dynamics, constraint, n_transitions=200, key=None):
    """Collect GP training data for 2D system."""
    if key is None:
        key = jax.random.key(0)

    X_list, Y_list = [], []
    x = jnp.array([-0.5, 0.0])  # start away from obstacle
    u_max = dynamics.u_max

    for _ in range(n_transitions):
        key, u_key = jax.random.split(key)
        u = jnp.array([jax.random.uniform(u_key, (), minval=-u_max, maxval=u_max)])

        x_next = dynamics.step(x, u)
        # Residual: (x' - x)/dt - f(x) - g(x)u
        residual = (x_next - x) / dynamics.dt - dynamics.f(x) - (dynamics.g(x) @ u).squeeze()

        X_list.append(x)
        Y_list.append(residual)

        # Reset if too far
        if jnp.abs(x_next[0]) > 3.0 or jnp.abs(x_next[1]) > 3.0:
            key, reset_key = jax.random.split(key)
            x = jnp.array([-1.0, 0.0]) + 0.5 * jax.random.normal(reset_key, (2,))
        else:
            x = x_next

    return jnp.stack(X_list), jnp.stack(Y_list)


def run_nominal_trajectory(dynamics, constraint, x_init, n_steps=500,
                            u_policy=None, key=None):
    """Run trajectory with nominal dynamics and given policy."""
    if key is None:
        key = jax.random.key(0)
    x = x_init
    traj = [x]
    h_vals = [float(constraint.h(x))]

    for t in range(n_steps):
        if u_policy is not None:
            key, u_key = jax.random.split(key)
            u = u_policy(x, u_key)
        else:
            u = jnp.array([0.0])

        x = dynamics.step(x, u)
        traj.append(x)
        h_vals.append(float(constraint.h(x)))

    return jnp.stack(traj), jnp.array(h_vals)


def run_robust_trajectory(dynamics, constraint, hocbf, qp_solver,
                          x_init, n_steps=500, key=None):
    """Run trajectory with QP safety filter."""
    if key is None:
        key = jax.random.key(0)
    x = x_init
    traj = [x]
    h_vals = [float(constraint.h(x))]

    for t in range(n_steps):
        # Nominal control: drift toward origin
        u_rl = jnp.array([-0.1 * x[0]])

        # QP safety filter
        A, b = hocbf.qp_matrices(x)
        u_safe, _ = qp_solver.solve_with_rl_action(u_rl, A, b, differentiable=False)

        x = dynamics.step(x, u_safe)
        traj.append(x)
        h_vals.append(float(constraint.h(x)))

    return jnp.stack(traj), jnp.array(h_vals)


def compute_safe_boundary(dynamics, constraint, gp=None, epsilon_kappa=1.0,
                          n_grid=100):
    """Compute h=0 and h=epsilon contours on a grid."""
    pos_range = np.linspace(-1.0, 2.0, n_grid)
    vel_range = np.linspace(-1.5, 1.5, n_grid)
    POS, VEL = np.meshgrid(pos_range, vel_range)

    h_grid = np.zeros_like(POS)
    eps_grid = np.zeros_like(POS)

    for i in range(n_grid):
        for j in range(n_grid):
            x = jnp.array([POS[i, j], VEL[i, j]])
            h_grid[i, j] = float(constraint.h(x))

            if gp is not None:
                hocbf = RobustHOCBF(
                    h_fn=constraint.h, f_fn=dynamics.f,
                    g_fn=dynamics.g, relative_degree=2,
                    k_gains=[1.0, 1.0], gp_residual=gp,
                    u_max=dynamics.u_max, epsilon_kappa=epsilon_kappa)
                eps_grid[i, j] = float(hocbf.compute_epsilon(x))

    return POS, VEL, h_grid, eps_grid


def run_method_comparison(dynamics, constraint, gp, qp_solver,
                          scenarios, n_steps=500, n_trials=5, key=None):
    """Compare violation rates across methods for each scenario.

    Returns dict: method_name → scenario_name → violation_count
    """
    if key is None:
        key = jax.random.key(42)

    results = {}

    # Method 1: Pure PPO (no safety)
    method = 'PPO'
    results[method] = {}
    for scenario_name in scenarios:
        total_violations = 0
        for trial in range(n_trials):
            key, trial_key = jax.random.split(key)
            x_init = jnp.array([-0.5, 0.3]) + 0.2 * jax.random.normal(trial_key, (2,))

            if scenario_name != 'nominal':
                dyn = UncertainDoubleIntegratorDynamics(
                    dt=dynamics.dt, u_max=dynamics.u_max,
                    uncertainty_scenario=scenario_name)
            else:
                dyn = dynamics

            model = ActorCritic(n_obs=2, n_act=1, hidden_dim=32, rngs=nnx.Rngs(trial))
            traj, h_vals = run_nominal_trajectory(dyn, constraint, x_init, n_steps,
                                                   key=trial_key)
            total_violations += int(jnp.sum(h_vals < 0))
        results[method][scenario_name] = total_violations / n_trials

    # Method 2: HOCBF (no GP)
    method = 'HOCBF'
    results[method] = {}
    hocbf_nom = HOCBF(h_fn=constraint.h, f_fn=dynamics.f,
                      g_fn=dynamics.g, relative_degree=2,
                      k_gains=[1.0, 1.0])
    for scenario_name in scenarios:
        total_violations = 0
        for trial in range(n_trials):
            key, trial_key = jax.random.split(key)
            x_init = jnp.array([-0.5, 0.3]) + 0.2 * jax.random.normal(trial_key, (2,))

            if scenario_name != 'nominal':
                dyn = UncertainDoubleIntegratorDynamics(
                    dt=dynamics.dt, u_max=dynamics.u_max,
                    uncertainty_scenario=scenario_name)
            else:
                dyn = dynamics

            traj, h_vals = run_robust_trajectory(
                dyn, constraint, hocbf_nom, qp_solver, x_init, n_steps,
                key=trial_key)
            total_violations += int(jnp.sum(h_vals < 0))
        results[method][scenario_name] = total_violations / n_trials

    # Method 3: Robust HOCBF (with GP, kappa=1.0)
    method = 'Robust-HOCBF'
    results[method] = {}
    hocbf_rob = RobustHOCBF(
        h_fn=constraint.h, f_fn=dynamics.f,
        g_fn=dynamics.g, relative_degree=2,
        k_gains=[1.0, 1.0], gp_residual=gp,
        u_max=dynamics.u_max, epsilon_kappa=1.0)
    for scenario_name in scenarios:
        total_violations = 0
        for trial in range(n_trials):
            key, trial_key = jax.random.split(key)
            x_init = jnp.array([-0.5, 0.3]) + 0.2 * jax.random.normal(trial_key, (2,))

            if scenario_name != 'nominal':
                dyn = UncertainDoubleIntegratorDynamics(
                    dt=dynamics.dt, u_max=dynamics.u_max,
                    uncertainty_scenario=scenario_name)
            else:
                dyn = dynamics

            traj, h_vals = run_robust_trajectory(
                dyn, constraint, hocbf_rob, qp_solver, x_init, n_steps,
                key=trial_key)
            total_violations += int(jnp.sum(h_vals < 0))
        results[method][scenario_name] = total_violations / n_trials

    # Method 4: RoCBF-Net (practical kappa)
    method = 'RoCBF-Net'
    results[method] = {}
    hocbf_rocbf = RobustHOCBF(
        h_fn=constraint.h, f_fn=dynamics.f,
        g_fn=dynamics.g, relative_degree=2,
        k_gains=[1.0, 1.0], gp_residual=gp,
        u_max=dynamics.u_max, epsilon_kappa=0.1)
    for scenario_name in scenarios:
        total_violations = 0
        for trial in range(n_trials):
            key, trial_key = jax.random.split(key)
            x_init = jnp.array([-0.5, 0.3]) + 0.2 * jax.random.normal(trial_key, (2,))

            if scenario_name != 'nominal':
                dyn = UncertainDoubleIntegratorDynamics(
                    dt=dynamics.dt, u_max=dynamics.u_max,
                    uncertainty_scenario=scenario_name)
            else:
                dyn = dynamics

            traj, h_vals = run_robust_trajectory(
                dyn, constraint, hocbf_rocbf, qp_solver, x_init, n_steps,
                key=trial_key)
            total_violations += int(jnp.sum(h_vals < 0))
        results[method][scenario_name] = total_violations / n_trials

    return results


def main():
    """Generate 2D validation data and plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== 2D Double Integrator Validation ===\n", flush=True)

    dynamics, constraint = setup_2d_system(dt=0.01, u_max=1.0)
    qp_solver = DifferentiableQP()

    # GP pre-training
    print("Pre-training GP on 2D residuals...", flush=True)
    key = jax.random.key(42)
    key, data_key = jax.random.split(key)
    X, Y = collect_gp_data_2d(dynamics, constraint, n_transitions=500, key=data_key)
    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(X, Y)
    print(f"  GP fitted on {gp.n_training_points} points", flush=True)

    output_dir = 'results/phase4/figures/'
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Phase portrait with safe boundaries
    print("Computing safe boundary contours...", flush=True)
    POS, VEL, h_grid, eps_grid = compute_safe_boundary(
        dynamics, constraint, gp, epsilon_kappa=0.1, n_grid=80)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # h=0 contour (nominal safe boundary)
    ax.contour(POS, VEL, h_grid, levels=[0], colors='blue', linewidths=2,
               linestyles='--')
    # h=epsilon contour (robust safe boundary)
    if np.max(eps_grid) > 0:
        robust_h = h_grid - eps_grid
        ax.contour(POS, VEL, robust_h, levels=[0], colors='red', linewidths=2,
                   linestyles='-')

    # Run example trajectories
    key, traj_key = jax.random.split(key)
    x_init = jnp.array([0.0, 0.5])

    # Nominal trajectory
    traj_nom, h_nom = run_nominal_trajectory(dynamics, constraint, x_init, 300)
    ax.plot(traj_nom[:, 0], traj_nom[:, 1], 'b-', alpha=0.6, label='Nominal')

    # HOCBF trajectory
    hocbf_nom = HOCBF(h_fn=constraint.h, f_fn=dynamics.f,
                      g_fn=dynamics.g, relative_degree=2, k_gains=[1.0, 1.0])
    traj_hocbf, h_hocbf = run_robust_trajectory(
        dynamics, constraint, hocbf_nom, qp_solver, x_init, 300)
    ax.plot(traj_hocbf[:, 0], traj_hocbf[:, 1], 'g-', alpha=0.6, label='HOCBF')

    # Robust HOCBF trajectory
    hocbf_rob = RobustHOCBF(h_fn=constraint.h, f_fn=dynamics.f,
                             g_fn=dynamics.g, relative_degree=2,
                             k_gains=[1.0, 1.0], gp_residual=gp,
                             u_max=dynamics.u_max, epsilon_kappa=0.1)
    traj_rob, h_rob = run_robust_trajectory(
        dynamics, constraint, hocbf_rob, qp_solver, x_init, 300)
    ax.plot(traj_rob[:, 0], traj_rob[:, 1], 'r-', alpha=0.6, label='RoCBF-Net')

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase Portrait: Safe Boundaries & Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}phase_portrait.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}phase_portrait.png", flush=True)

    # Plot 2: Epsilon vs state
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    pos_slice = np.linspace(-1.0, 2.0, 200)
    eps_slice = []
    for p in pos_slice:
        x = jnp.array([p, 0.0])
        eps_slice.append(float(hocbf_rob.compute_epsilon(x)))
    ax.plot(pos_slice, eps_slice, 'r-', linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Obstacle center')
    ax.set_xlabel('Position')
    ax.set_ylabel('ε(x)')
    ax.set_title('Robustness Margin ε(x) along Position Axis (v=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f'{output_dir}epsilon_vs_state.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}epsilon_vs_state.png", flush=True)

    # Plot 3: Violation comparison
    print("Running method comparison...", flush=True)
    scenarios = ['nominal', 'damping', 'periodic', 'coupled', 'nonlinear']
    scenario_labels = ['Nominal', 'S1:Damping', 'S2:Periodic', 'S3:Coupled', 'S4:Nonlinear']
    results = run_method_comparison(dynamics, constraint, gp, qp_solver,
                                    scenarios, n_steps=300, n_trials=3, key=key)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    methods = list(results.keys())
    x_pos = np.arange(len(scenario_labels))
    width = 0.2

    for i, method in enumerate(methods):
        vals = [results[method].get(s, 0) for s in scenarios]
        ax.bar(x_pos + i * width, vals, width, label=method, alpha=0.8)

    ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel('Avg. Violations')
    ax.set_title('Constraint Violations: 2D Double Integrator')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(f'{output_dir}violation_comparison_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}violation_comparison_2d.png", flush=True)

    print("\n2D validation complete.", flush=True)


if __name__ == "__main__":
    main()
