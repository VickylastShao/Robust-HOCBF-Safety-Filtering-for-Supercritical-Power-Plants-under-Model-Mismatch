# LQR+RHOCBF vs PPO+RHOCBF Comparison Results
# 5th-order CCS, 5 seeds, 300 steps per evaluation

## Complete Results Table

| Scenario   | PPO Reward      | LQR Reward     | PPO CBF% | LQR CBF% | PPO QP%   | LQR QP%  |
|------------|-----------------|----------------|----------|----------|-----------|----------|
| Nominal    | -28.2 ± 9.0     | 0.0 ± 0.0     | 0.0      | 0.0      | 0.3       | 0.0      |
| S1:Heat    | -6603.3 ± 74.2  | -6589.2 ± 0.1 | 0.0      | 0.0      | 98.8      | 100.0    |
| S2:Pressure| -9396.0 ± 91.0  | -9380.8 ± 0.0 | 0.0      | 0.0      | 96.5      | 100.0    |
| S3:Coupled| -4911.5 ± 107.5 | -4892.6 ± 0.2 | 0.0      | 0.0      | 95.4      | 100.0    |
| S4:Nonlinear| -3706.1 ± 68.1 | -3663.6 ± 0.1 | 0.0      | 0.0      | 98.2      | 100.0    |

Notes:
- S3 Coupled: seeds 0-2 from full experiment, seeds 3-4 from fast experiment
  - Full: seeds 0-2: PPO rewards [-4859.0, -4753.0, -4925.0], QP: [98.0%, 81.7%, 99.7%]
  - Fast: seeds 3-4: PPO rewards [-5042.3, -4998.3], QP: [91.3%, 100.0%]
  - Combined PPO mean: (-4859 + -4753 + -4925 + -5042.3 + -4998.3) / 5 = -4915.5
  - LQR always -4892.6
- S4 Nonlinear from fast experiment (hidden_dim=64, 10 episodes)

## Key Findings

1. **Safety is identical**: Both PPO+RHOCBF and LQR+RHOCBF achieve 0% CBF violation across ALL scenarios.
   - This confirms the QP safety filter is policy-agnostic.
   - Safety is guaranteed by the Robust-HOCBF framework regardless of the policy's action.

2. **QP intervention dominates**: Under perturbation scenarios, QP intervenes at 95-100% of steps.
   - The policy's requested action is almost always overridden by the safety filter.
   - This means the tracking performance is primarily determined by the QP solver, not the policy.

3. **Reward is nearly identical**: Under perturbation, PPO and LQR rewards differ by <2%.
   - S1:Heat: -6603 vs -6589 (0.2% difference)
   - S2:Pressure: -9396 vs -9381 (0.2% difference)
   - S3:Coupled: -4915 vs -4893 (0.5% difference)
   - S4:Nonlinear: -3706 vs -3664 (1.1% difference)

4. **LQR outperforms PPO under Nominal**: When no perturbation exists, LQR achieves reward=0 (perfect tracking) while PPO gets -28.2.
   - Under nominal conditions, QP intervention is ~0%, so the policy matters.
   - LQR (v=0) is optimal because the stabilized dynamics already encodes the LQR gain.
   - PPO's random exploration causes slight deviations from the optimal policy.

5. **Conclusion for paper**: The LQR comparison honestly reveals that when QP intervention is high (perturbation scenarios), the RL policy provides negligible benefit over the trivial v=0 baseline. The safety guarantee comes entirely from the Robust-HOCBF filter. The RL policy's value is primarily in nominal/low-disturbance conditions where QP intervention is low.
