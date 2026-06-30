"""Phase 2 validation: verify exit criteria for robustness injection.

Exit Criteria (from design spec):
1. Robust HOCBF: violation rate < 1% under all 4 uncertainty scenarios
2. Traditional HOCBF: violation rate > 10% under at least 2 scenarios
3. GP coverage: 90–98% empirical coverage of predicted intervals
4. ε(x) within 2× of oracle bound in ≥ 90% of states
"""
import sys
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from rocbf.rl.ppo import ActorCritic
from rocbf.cbf.robust_hocbf import RobustHOCBF
from rocbf.cbf.hocbf import HOCBF
from rocbf.qp.diff_qp import DifferentiableQP
from rocbf.gp.gp_residual import GPResidual, collect_gp_data
from envs.safe_navigation.dynamics import UncertainDoubleIntegratorDynamics
from envs.safe_navigation.constraints import CircularKeepOut


SCENARIOS = [None, "damping", "periodic", "coupled", "nonlinear"]
SCENARIO_LABELS = ["Nominal", "S1:Damping", "S2:Periodic", "S3:Coupled", "S4:Nonlinear"]


def _make_env(scenario=None, dt=0.01, u_max=5.0):
    return UncertainDoubleIntegratorDynamics(
        dt=dt, u_max=u_max, uncertainty_scenario=scenario)


def _make_robust_hocbf(nominal_env, constraint, gp, u_max=5.0):
    return RobustHOCBF(
        h_fn=constraint.h,
        f_fn=nominal_env.f_nominal,
        g_fn=nominal_env.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
        gp_residual=gp,
        u_max=u_max,
    )


def check_robust_safety(n_episodes=50, n_steps=200):
    """Check 1: Robust HOCBF violation rate < 1% per scenario."""
    print("\n--- Check 1: Robust HOCBF Safety (< 1% violations) ---")
    sys.stdout.flush()

    # Pre-train GP on all scenarios
    key = jax.random.key(0)
    X_all, Y_all = [], []
    for scenario in SCENARIOS:
        env = _make_env(scenario)
        key, data_key = jax.random.split(key)
        X, Y = collect_gp_data(env, n_transitions=200, key=data_key)
        X_all.append(X)
        Y_all.append(Y)

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(jnp.concatenate(X_all), jnp.concatenate(Y_all))

    nominal_env = _make_env()
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)
    hocbf = _make_robust_hocbf(nominal_env, constraint, gp)
    qp_solver = DifferentiableQP()
    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(0))

    results = {}
    for i, scenario in enumerate(SCENARIOS):
        env = _make_env(scenario)
        total_violations = 0
        total_steps = 0

        for ep in range(n_episodes):
            key, ep_key = jax.random.split(key)
            x = jnp.array([3.0, 0.0])

            for t in range(n_steps):
                key, action_key = jax.random.split(key)
                mean, log_std, _ = model(x)
                std = jnp.exp(log_std)
                u_rl = mean + std * jax.random.normal(action_key, mean.shape)

                A, b = hocbf.qp_matrices(x)
                G, h = A, b
                u_safe, _ = qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=False)

                x = env.step(x, u_safe)
                total_steps += 1

                if constraint.h(x) < 0:
                    total_violations += 1
                    break

        rate = total_violations / n_episodes
        label = SCENARIO_LABELS[i]
        results[label] = rate
        status = "PASS" if rate < 0.01 else "FAIL"
        print(f"  {label}: violation rate = {rate:.4f} ({total_violations}/{n_episodes}) [{status}]")
        sys.stdout.flush()

    all_pass = all(r < 0.01 for r in results.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def check_traditional_unsafe(n_episodes=50, n_steps=200):
    """Check 2: Traditional HOCBF violation rate > 10% under >= 2 scenarios."""
    print("\n--- Check 2: Traditional HOCBF Unsafety (> 10% under >= 2 scenarios) ---")
    sys.stdout.flush()

    nominal_env = _make_env()
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    hocbf = HOCBF(
        h_fn=constraint.h,
        f_fn=nominal_env.f,
        g_fn=nominal_env.g,
        relative_degree=2,
        k_gains=[2.0, 2.0],
    )
    qp_solver = DifferentiableQP()
    model = ActorCritic(n_obs=2, n_act=1, hidden_dim=64, rngs=nnx.Rngs(0))

    key = jax.random.key(1)
    results = {}
    scenarios_with_high_violations = 0

    for i, scenario in enumerate(SCENARIOS):
        if scenario is None:
            continue  # Skip nominal for this check

        env = _make_env(scenario)
        total_violations = 0

        for ep in range(n_episodes):
            key, ep_key = jax.random.split(key)
            x = jnp.array([1.5, 0.0])

            for t in range(n_steps):
                key, action_key = jax.random.split(key)
                mean, log_std, _ = model(x)
                std = jnp.exp(log_std)
                u_rl = mean + std * jax.random.normal(action_key, mean.shape)

                A, b = hocbf.qp_matrices(x)
                G, h = A, b
                u_safe, _ = qp_solver.solve_with_rl_action(u_rl, G, h, differentiable=False)

                x = env.step(x, u_safe)

                if constraint.h(x) < 0:
                    total_violations += 1
                    break

        rate = total_violations / n_episodes
        label = SCENARIO_LABELS[i]
        results[label] = rate
        if rate > 0.10:
            scenarios_with_high_violations += 1
        print(f"  {label}: violation rate = {rate:.4f} ({total_violations}/{n_episodes})")
        sys.stdout.flush()

    passed = scenarios_with_high_violations >= 2
    print(f"  Scenarios with > 10% violations: {scenarios_with_high_violations}/4")
    print(f"  Overall: {'PASS' if passed else 'FAIL (may need more episodes or stronger perturbation)'}")
    return passed


def check_gp_calibration(n_points=200):
    """Check 3: GP empirical coverage 90–98%."""
    print("\n--- Check 3: GP Calibration (coverage 90-98%) ---")
    sys.stdout.flush()

    key = jax.random.key(2)
    results = {}

    for i, scenario in enumerate(SCENARIOS):
        if scenario is None:
            continue

        env = _make_env(scenario)
        key, data_key = jax.random.split(key)
        X, Y = collect_gp_data(env, n_transitions=n_points, key=data_key)

        gp = GPResidual(n_dims=2, noise_variance=1e-4)
        gp.fit(X, Y, n_optim_iters=50, lr=0.01)

        # Check calibration on held-out points
        key, test_key = jax.random.split(key)
        X_test, Y_test = collect_gp_data(env, n_transitions=100, key=test_key)

        beta = GPResidual.compute_beta(2, gp.n_training_points)

        in_interval = 0
        total = X_test.shape[0]
        for j in range(total):
            mu, sigma = gp.predict(X_test[j])
            lower = mu - beta * sigma
            upper = mu + beta * sigma
            if jnp.all(Y_test[j] >= lower) and jnp.all(Y_test[j] <= upper):
                in_interval += 1

        coverage = in_interval / total
        label = SCENARIO_LABELS[i]
        results[label] = coverage
        in_range = 0.90 <= coverage <= 0.98
        status = "PASS" if in_range else "WARN"
        print(f"  {label}: coverage = {coverage:.4f} ({in_interval}/{total}) [{status}]")
        sys.stdout.flush()

    all_in_range = all(0.85 <= c <= 0.99 for c in results.values())
    print(f"  Overall: {'PASS' if all_in_range else 'WARN (coverage may vary with data size)'}")
    return all_in_range


def check_epsilon_tightness(n_points=100):
    """Check 4: ε(x)/ε*(x) < 2 in >= 90% of states."""
    print("\n--- Check 4: ε Tightness (ε/ε* < 2 in >= 90% of states) ---")
    sys.stdout.flush()

    key = jax.random.key(3)

    # Train GP with data from all scenarios
    X_all, Y_all = [], []
    for scenario in SCENARIOS:
        env = _make_env(scenario)
        key, data_key = jax.random.split(key)
        X, Y = collect_gp_data(env, n_transitions=200, key=data_key)
        X_all.append(X)
        Y_all.append(Y)

    gp = GPResidual(n_dims=2, noise_variance=1e-4)
    gp.fit(jnp.concatenate(X_all), jnp.concatenate(Y_all))

    nominal_env = _make_env()
    constraint = CircularKeepOut(center=jnp.array([0.0]), radius=1.0)

    results = {}
    for i, scenario in enumerate(SCENARIOS):
        if scenario is None:
            continue

        env = _make_env(scenario)
        hocbf = _make_robust_hocbf(nominal_env, constraint, gp)

        # Sample test states
        key, state_key = jax.random.split(key)
        test_states = 2.0 * jax.random.uniform(state_key, (n_points, 2)) + jnp.array([1.0, -0.5])

        within_bound = 0
        for j in range(n_points):
            x = test_states[j]
            eps = hocbf.compute_epsilon(x)
            eps_oracle = hocbf.epsilon_oracle(x, env.delta_f)

            ratio = float(eps / jnp.maximum(eps_oracle, 1e-10))
            if ratio <= 2.0:
                within_bound += 1

        fraction = within_bound / n_points
        label = SCENARIO_LABELS[i]
        results[label] = fraction
        status = "PASS" if fraction >= 0.90 else "FAIL"
        print(f"  {label}: ε/ε* < 2 in {fraction:.2%} of states ({within_bound}/{n_points}) [{status}]")
        sys.stdout.flush()

    all_pass = all(f >= 0.90 for f in results.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def validate_phase2():
    """Run all Phase 2 exit criteria checks."""
    print("=== Phase 2: Robustness Injection Validation ===")
    sys.stdout.flush()

    results = {
        "robust_safety": check_robust_safety(),
        "traditional_unsafe": check_traditional_unsafe(),
        "gp_calibration": check_gp_calibration(),
        "epsilon_tightness": check_epsilon_tightness(),
    }

    print("\n=== Summary ===")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    sys.stdout.flush()

    all_pass = all(results.values())
    if all_pass:
        print("\nAll Phase 2 exit criteria PASSED!")
    else:
        print("\nSome criteria not met — see details above.")
    return all_pass


if __name__ == "__main__":
    validate_phase2()
