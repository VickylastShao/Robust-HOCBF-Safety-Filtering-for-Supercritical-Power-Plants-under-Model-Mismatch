# RoCBF-Net Paper Wiki — Index

## Papers

| Page | Authors | Year | One-Line Summary |
|------|---------|------|-----------------|
| [xiao2019](papers/xiao2019.md) | Xiao & Belta | 2019 | Defines HOCBF via ψ-chain recursion for systems with high relative degree safety constraints |
| [barriernet](papers/barriernet.md) | Xiao, Hasani, Li & Rus | 2021 | Differentiable HOCBF-QP as a trainable NN layer with environment-adaptive penalty parameters |
| [ma2022](papers/ma2022.md) | Ma et al. | 2022 | Differentiable ECBF-QP with learned class-K gains (α-net) for generalization to novel environments |
| [cohen2022](papers/cohen2022.md) | Cohen & Belta | 2022 | HO-RaCBF extends HOCBF to parametric uncertain systems with robustness buffer via concurrent learning |
| [das2024](papers/das2024.md) | Das, Wei & Burdick | 2024 | Robust CBF with uncertainty estimator for unmodeled dynamics, extends to HOCBF via Corollary 1 |
| [rasmussen2006](papers/rasmussen2006.md) | Rasmussen & Williams | 2006 | GP regression textbook — posterior formulas, kernels, marginal likelihood, sparse approximations |
| [sarkka2013a](papers/sarkka2013a.md) | Särkkä & Hartikainen | 2013 | GP→infinite-dimensional Kalman filter, O(T³)→O(T) for spatio-temporal GP regression |
| [sarkka2013b](papers/sarkka2013b.md) | Särkkä, Solin & Hartikainen | 2013 | Journal extension: systematic GP↔state-space conversion with spectral factorization procedure |
| [guerrier2024](papers/guerrier2024.md) | Guerrier, Fouad & Beltrame | 2024 | Survey of CBF+RL: confirms no prior work combines GP robustness + HOCBF + differentiable QP |
| [choi2021](papers/choi2021.md) | Choi et al. | 2021 | CBVF merges HJ reachability with CBF, robust safety via value functions (curse of dimensionality) |

## Concepts

| Page | Description | Source Papers |
|------|-------------|---------------|
| [hocbf](concepts/hocbf.md) | High-Order CBF: ψ-chain, constraints, class-K functions | xiao2019, cohen2022, das2024 |
| [differentiable-qp](concepts/differentiable-qp.md) | Differentiable QP layers: KKT gradients, cvxpylayers | barriernet, ma2022 |
| [gp-state-space](concepts/gp-state-space.md) | GP regression as state-space model for O(n) online inference | rasmussen2006, sarkka2013a, sarkka2013b |
| [robust-cbf](concepts/robust-cbf.md) | Robust CBF methods: parametric, estimator-based, GP-based | cohen2022, das2024, choi2021 |

## Comparisons

| Page | Description |
|------|-------------|
| [robustness-approaches](comparisons/robustness-approaches.md) | Cohen 2022 vs Das 2024 vs Choi 2021: three routes to robust safety |
| [diff-qp-architectures](comparisons/diff-qp-architectures.md) | BarrierNet vs Ma 2022: two differentiable QP layer designs |
