# Wiki Log

## [2026-05-19] ingest | Batch ingest of 10 papers

Papers processed (parallel agent batch):
1. xiao2019 — HOCBF (Xiao & Belta 2019)
2. barriernet — BarrierNet (Xiao et al. 2021)
3. ma2022 — Diff-ECBF-QP (Ma et al. 2022)
4. cohen2022 — HO-RaCBF (Cohen & Belta 2022)
5. das2024 — Robust CBF with Uncertainty Estimation (Das et al. 2024)
6. rasmussen2006 — GPML textbook (Rasmussen & Williams 2006)
7. sarkka2013a — GP-Kalman (Särkkä & Hartikainen 2013)
8. sarkka2013b — GP-Bayesian Filtering (Särkkä et al. 2013)
9. guerrier2024 — CBF+RL Survey (Guerrier et al. 2024)
10. choi2021 — CBVF (Choi et al. 2021)

Created: 10 paper pages, index.md, log.md

## [2026-05-19] ingest | Concept and comparison pages

Created:
- concepts/hocbf.md — ψ-chain, m=2 linear case, robustness extensions
- concepts/differentiable-qp.md — KKT gradients, BarrierNet vs Ma 2022 approaches
- concepts/gp-state-space.md — GP→Kalman conversion, Matérn kernel state-space forms
- concepts/robust-cbf.md — Cohen 2022 vs Das 2024 vs Choi 2021 vs RoCBF-Net
- comparisons/robustness-approaches.md — Uncertainty model, guarantee type, computational cost
- comparisons/diff-qp-architectures.md — BarrierNet vs Ma 2022 QP layer design

Key finding from [[guerrier2024]]: Survey confirms no prior work combines GP robustness + HOCBF + differentiable QP → RoCBF-Net novelty validated
