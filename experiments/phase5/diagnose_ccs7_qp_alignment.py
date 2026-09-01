"""Diagnose HOCBF-QP alignment on the seven-state benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp

from experiments.phase5.methods_7th import (
    DEVIATION_COMPONENT_BOUND,
    _make_ccs_env_7th,
    _make_hocbf_7th,
)
from experiments.phase5.run_drift_only_fixed_proposal import (
    CONDITION_SCENARIO_MAP,
    proposal_action,
    sample_initial_state,
)
from rocbf.qp.diff_qp import DifferentiableQP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--load-ratio", type=float, default=0.66)
    parser.add_argument("--scenario-scale", type=float, default=0.005)
    parser.add_argument("--hocbf-gain", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenario = CONDITION_SCENARIO_MAP[args.condition]
    dynamics, constraints = _make_ccs_env_7th(
        args.load_ratio, scenario, scenario_scale=args.scenario_scale)
    x0, u0 = dynamics.equilibrium(args.load_ratio)
    safety = _make_hocbf_7th(
        dynamics, constraints, u0,
        k_gains=(args.hocbf_gain, args.hocbf_gain))
    qp = DifferentiableQP(v_max=DEVIATION_COMPONENT_BOUND)
    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    x, key = sample_initial_state(
        dynamics, constraints, init_key, admissibility_layer=safety)
    trace = []

    for step in range(args.steps):
        key, action_key = jax.random.split(key)
        proposed = proposal_action(step, 0, args.seed, action_key)
        A, b = safety.qp_matrices(x)
        applied, _, info = qp.solve_with_rl_action(
            proposed, A, b, differentiable=False,
            fallback_v=proposed, feasibility_tol=1e-6,
            return_info=True)
        next_x = dynamics.step_stabilized(x, applied)
        before = constraints.check_all(x)
        after = constraints.check_all(next_x)
        trace.append({
            "step": step,
            "state": jnp.asarray(x).tolist(),
            "proposed": jnp.asarray(proposed).tolist(),
            "applied": jnp.asarray(applied).tolist(),
            "intervened": bool(jnp.any(jnp.abs(applied - proposed) > 1e-3)),
            "solver_success": bool(info["success"]),
            "solver_message": str(info["solver_message"]),
            "max_normalized_qp_residual": float(
                info["max_normalized_constraint_residual"]),
            "min_margin_before": min(before.values()),
            "min_margin_after": min(after.values()),
            "margins_before": before,
            "margins_after": after,
            "A": jnp.asarray(A).tolist(),
            "b": jnp.asarray(b).tolist(),
            "continuous_constraint_residual": jnp.asarray(A @ applied - b).tolist(),
        })
        x = next_x

    payload = {
        "condition": args.condition,
        "seed": args.seed,
        "load_ratio": args.load_ratio,
        "scenario_scale": args.scenario_scale,
        "hocbf_gain": args.hocbf_gain,
        "first_violating_step": next(
            (row["step"] for row in trace if row["min_margin_after"] < -0.01),
            None),
        "trace": trace,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "output": str(output),
        "first_violating_step": payload["first_violating_step"],
        "interventions": sum(row["intervened"] for row in trace),
        "solver_failures": sum(not row["solver_success"] for row in trace),
        "minimum_margin": min(row["min_margin_after"] for row in trace),
    }, indent=2))


if __name__ == "__main__":
    main()
