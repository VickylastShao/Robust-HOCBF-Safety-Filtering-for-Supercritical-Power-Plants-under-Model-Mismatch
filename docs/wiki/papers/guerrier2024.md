---
title: CBF+RL Survey — Learning CBFs and their Application in RL
authors: [Maeva Guerrier, Hassan Fouad, Giovanni Beltrame]
year: 2024
venue: arXiv (2404.16879)
tags: [survey, cbf, reinforcement-learning, safe-rl, learning-cbf]
sources: [guerrier2024]
updated: 2026-05-19
---

## One-Line Summary

A survey reviewing safe reinforcement learning (SRL) methods that use Control Barrier Functions, with particular attention to data-driven techniques for learning CBFs from demonstrations, rollouts, and priors, and their integration into RL training and deployment pipelines.

## Problem Setting (scope of the survey)

The survey addresses two intertwined problems:

1. **Safety in RL**: Standard RL agents explore by taking potentially unsafe actions. In safety-critical domains (human-robot interaction, autonomous driving), this is unacceptable. The survey examines how CBFs can provide hard or probabilistic safety guarantees during both RL training and deployment.

2. **Learning CBFs from data**: Hand-crafting CBFs requires significant domain knowledge and often yields conservative safe sets. The survey reviews methods that construct or refine CBFs using machine learning, expert demonstrations, or data-driven priors, making CBFs more practical for RL integration.

The survey organizes the landscape along two axes:
- **Type of safety constraint**: soft (no guarantees, encourage safety), hard (deterministic guarantees via safety filters), probabilistic (guarantees within a confidence bound).
- **Model dependence**: model-based (known or learned dynamics) vs. model-free (no dynamics model for the RL policy, though a model may be used solely for the safety filter).

## Taxonomy of Approaches

The survey categorizes CBF+RL methods into a hierarchy:

### Level 1: Safety Constraint Type
- **Soft constraints**: Reward shaping, CMDP-based policy optimization (CPO, RCPO), safety critics, learning from demonstration (ConBat). No guarantees during training; safety encouraged asymptotically.
- **Hard constraints**: Safety shields (LTL-based, learned shields) and CBF safety filters that deterministically prevent unsafe actions. Require some system model.
- **Probabilistic constraints**: CBF+GP frameworks where uncertainty is modeled statistically, yielding safety guarantees with high probability.

### Level 2: CBF Integration Mechanism
- **Safety shields**: Monitor and override actions before execution. Can be LTL-based, learned binary classifiers, or CBF-based.
- **CBF safety filters**: Solve a QP at each step: $\min_u \|u - k(x)\|^2$ s.t. $L_f h + L_g h \, u \geq -\alpha(h(x))$. The CBF constraint acts as a hard or probabilistic filter on the RL policy's proposed action.

### Level 3: CBF Construction Method
- **Hand-crafted CBFs (HCBF)**: Manually designed based on domain knowledge. Conservative but reliable.
- **Learning from demonstrations (LfD)**: CBFs extracted from expert safe/unsafe trajectory data.
- **RL-based CBF learning**: Inverse RL, on-policy rollout collection, CLBFs via modified Bellman equations, CBFs from RL value functions.
- **GP-based CBF construction**: Gaussian Processes model unmodeled dynamics or directly construct CBFs (Gaussian CBFs), providing uncertainty-aware safety.
- **CBF priors / refinement**: Start from a known HCBF and expand its safe set (DDN-based increments), learn adaptive $\alpha$ functions, or trim an over-approximation to the true safe set.

## Key Categories

### 1. Soft Constraint SRL Methods
- **Risk-aware RL** (Kahn et al., 2017): Augments cost with probabilistic risk model; permits low-impact violations to learn risk prevention.
- **Safety critics** (Srinivasan et al., 2020): Pre-train a safety critic on unsafe states, then fine-tune a new policy using it.
- **CMDP approaches** (Achiam et al., 2017 / CPO; Tessler et al., 2018 / RCPO; Miryoosefi et al., 2019): Constrained policy optimization with surrogate bounds, Lagrange relaxation, or game-theoretic mixed strategies.
- **ConBat** (Meng et al., 2023): Learns a CBF-like safety critic from demonstrations to augment a PACT architecture. No formal guarantees; vulnerable to OOD data and local minima.
- **Sim-to-lab-to-real** (Hsu et al., 2022): Jointly trains backup + performance policies with a discriminator; fine-tunes in controlled environment before real deployment.

### 2. Hard and Probabilistic Constraint SRL Methods

#### 2a. Safety Shields
- **LTL shields** (Alshiekh et al., 2018): Discrete-state reactive system; estimates winning region; limited to discrete action/state spaces.
- **Multi-agent shields** (Elsayed-Aly et al., 2021): Centralized shield monitors all agents; factorized shields for scalability.
- **Learned shields** (Sibai et al., 2019): Binary safe/unsafe classifier replaces explicit reachability computation.
- **Shield SARSA** (Zhao et al., 2023): Shield removes unsafe actions from action set; SARSA re-samples.

#### 2b. CBF Safety Filters
- **CBF + GP** (Cheng et al., 2019): Nominal model + GP disturbance; probabilistic guarantees; filters RL actions during training; incorporates history of filtered actions to improve policy.
- **Robust CBF + RL** (Emam et al., 2022): Robust CBFs with disturbance estimates; minimal action adjustments; hard guarantees under bounded disturbance.
- **Disturbance Observer CBF (DOBCBF)** (Cheng et al., 2022/2023): High relative degree CBF with disturbance observer; predicts disturbance upper bound; filtered actions stored in replay buffer for policy training.
- **UTCBF** (Zhang et al., 2021): Uncertainty-Tolerant CBFs with GP; conservative early (large error bound) then relaxes as data accumulates; model-based RL with policy updates via constrained optimization.
- **GCBF** (Ma et al., 2021): Generalized CBF for high relative degree; model-based constrained policy optimization.
- **ECBF for RL** (Hailemichael et al., 2022): Exponential CBFs handle high relative degree; applied to energy-efficient driving.
- **GCBF+ / Multi-agent** (Zhang et al., 2024): Graph neural network parameterization of distributed CBF; unified loss for safety + goal-reaching.

### 3. CBF Construction Methods

#### 3a. RL-Based CBF Learning
- **Inverse RL for CBF** (Yang et al., 2022): Neural network CBF trained from safe/unsafe demonstrations; loss enforces CBF properties.
- **Decentralized neural barrier certificates** (Qin et al., 2021): On-policy data collection; permutation-invariant architecture; CBF-QP-inspired policy refinement.
- **CLBF via Bellman** (Du et al., 2023): Modified Bellman equation respects CLBF properties; actor-critic with CLBF-constrained value function.
- **RL value function as CBF** (Scukins & Ogren, 2021): Constructs CBF directly from RL value function; sparse rewards limit intermediate learning.
- **LiDAR-based CBF** (Srinivasan et al., 2020): Gaussian kernel NN + SVM from LiDAR data; safe/unsafe samples from environment sensing.

#### 3b. Learning CBF from Demonstrations
- **Incremental ZCBF** (Saveriano & Lee, 2019): Clusters unsafe states; fits linear-combination ZCBFs per cluster.
- **Expert trajectory CBF** (Robey et al., 2020): CBF learned from safe/unsafe trajectory data.
- **Safe-only demonstrations** (Lindemann et al., 2020): Hybrid system CBF from safe behaviors only; unsafe set defined implicitly outside demonstrations; constrained optimization with DNN.
- **ROCBF from demonstrations** (Lindemann et al., 2021): Extends above with model and state estimation errors; Robust Output CBF via constrained optimization.

#### 3c. GP-Based CBF Construction
- **Gaussian CBF** (Khan et al., 2020/2022): Posterior positive in high-confidence safe regions; online construction; safe set shapes CBF structure directly.
- **GP + SMT for CBF** (Jagtap et al., 2020): GP learns unknown dynamics; SMT solver finds valid CBF and control policy.
- **GP uncertainty in CBF** (Khan et al., 2021): Posterior variance from GP conditions the Gaussian CBF; QP-based safety filter.
- **Decentralized multi-robot GP-CBF** (Hu et al., 2023): Per-robot GP for individual uncertainty; decentralized robust CBF conditions in QP.
- **GP for safe set expansion** (Berkenkamp et al., 2016): Bayesian optimization to expand region of attraction.

#### 3d. CBF Priors and Refinement
- **DDN-based CBF expansion** (Dai et al., 2022): Starts with HCBF; trains a Deep Differential Network increment to expand safe set toward true boundary.
- **PER-based refinement** (Dai et al., 2023): Prioritized Experience Replay for better sample efficiency in CBF increment training.
- **Shrinking over-approximation** (Lee et al., 2023): Starts with bigger (non-invariant) set; trims via scaling/offset to converge to true safe set.
- **Adaptive $\alpha$ via GNN** (Gao et al., 2023): Learns the extended class-$\mathcal{K}$ function $\alpha(h)$ as a GNN policy; balances conservatism vs. feasibility.
- **Differentiable safety for ECBF** (Ma et al., 2022): Learns CBF constraint structure for high relative degree systems; differentiable formulation for generalization.
- **Local CBF for hybrid systems** (Yang et al., 2024): DP-based dynamic unsafe set computation; local CBFs for high-dimensional hybrid systems.
- **Iterative barrier certificates** (Luo & Ma, 2021): Iteratively learns barrier certificates with HCBF prior; limited data handling noted.
- **HJ-based CBF refinement** (Tonkens & Herbert, 2022): Hamilton-Jacobi reachability to refine invalid/conservative CBFs; requires known obstacle dynamics.
- **AlwaysSafe** (Simao et al., 2021): Zero training-time violations through progressive CBF enhancement.

## Comparison with RoCBF-Net

RoCBF-Net combines **HOCBF (High-Order CBF)** + **Gaussian Process robustness** + **differentiable QP**. Positioning within this survey's taxonomy:

| Aspect | Where RoCBF-Net fits | What has been done before | What is novel |
|--------|---------------------|--------------------------|---------------|
| **CBF type** | High-Order CBF (HOCBF) for high relative degree | ECBF (Hailemichael et al., 2022), GCBF (Ma et al., 2021), DOBCBF (Cheng et al., 2022) all address high relative degree | HOCBF formulation with explicit chain of derivatives differs from ECBF/GCBF; combination with GP and differentiable QP is new |
| **Uncertainty handling** | GP models disturbance/uncertainty, integrated into HOCBF constraints | GP+CBF extensively studied: Cheng et al. (2019), UTCBF (Zhang et al., 2021), Gaussian CBF (Khan et al., 2020/2022), GP-CBF (Jagtap et al., 2020), GP uncertainty (Khan et al., 2021) | GP integrated into HOCBF (not just basic CBF); GP uncertainty propagates through the high-order derivative chain |
| **QP formulation** | Differentiable QP layer embedded in neural network | Standard CBF-QP (Ames et al., 2019) is well-known but non-differentiable; Ma et al. (2022) learn differentiable safety-critical control for ECBF | Differentiable QP specifically for HOCBF + GP is novel; enables end-to-end gradient flow through the safety filter into policy training |
| **End-to-end training** | Policy and CBF parameters trained jointly via differentiable QP | Most works use CBF as external filter (not differentiable w.r.t. policy). Ma et al. (2022) is closest but uses ECBF without GP robustness. Qin et al. (2021) use CBF-QP-inspired refinement but not differentiable QP layer. | Joint training of policy + HOCBF + GP uncertainty model through differentiable QP is the key novelty |
| **Safe set learning** | Not applicable (HOCBF structure defined by constraints) | DDN expansion (Dai et al., 2022), shrinking over-approximation (Lee et al., 2023), Gaussian CBF (Khan et al., 2022) | RoCBF-Net does not learn the safe set; it starts from known constraints and focuses on robust enforcement |

**Key novelty claim**: The combination of (1) HOCBF for high relative degree, (2) GP-based uncertainty quantification propagated through the HOCBF chain, and (3) differentiable QP enabling end-to-end training, is not present in any single work surveyed. The closest prior is Ma et al. (2022) which has differentiable safety-critical control but uses ECBF without GP robustness modeling.

## Open Problems and Future Directions

The survey identifies several open problems, each representing an opportunity for RoCBF-Net:

1. **Sim2real gap and deployment certification**: Most safe RL methods are validated only in simulation. The survey notes that safety certification during deployment (not just training) is largely unaddressed. Only Hsu et al. (2022) tackles this for soft constraints; hard/probabilistic deployment certification remains open. *Opportunity: RoCBF-Net's GP-based robustness naturally accounts for model mismatch, making it a candidate for bridging sim2real in safety-critical deployment.*

2. **Generalizability and zero-shot performance**: Few works evaluate safe RL policies in environments different from training. The survey cites Emam et al. (2022) and a few others as exceptions. *Opportunity: RoCBF-Net's differentiable QP + GP framework could be evaluated for zero-shot transfer, as the GP adapts to new dynamics online.*

3. **Sample efficiency of CBF learning**: Methods that learn CBFs from data (DDN expansion, Gaussian CBFs, iterative certificates) require many samples. Only Dai et al. (2023) addresses sample efficiency via PER. *Opportunity: RoCBF-Net's end-to-end differentiable training could be more sample-efficient than separate CBF learning + policy training.*

4. **Conservatism vs. feasibility trade-off**: Fixed $\alpha$ functions in CBF constraints lead to conservatism or infeasibility. Gao et al. (2023) learn adaptive $\alpha$; Ma et al. (2022) learn differentiable constraint structure. *Opportunity: RoCBF-Net's differentiable QP allows tuning the entire safety pipeline end-to-end, potentially finding better trade-offs automatically.*

5. **Model-free safe RL with hard guarantees**: The survey observes that model-free RL methods generally only achieve soft constraints unless augmented with a separate model for safety. *Opportunity: RoCBF-Net's GP learns the model uncertainty online, effectively converting a model-free RL setting into one with probabilistic hard guarantees as data accumulates.*

6. **Updating invalid CBFs**: Most works assume a valid initial CBF. Tonkens & Herbert (2022) address refinement of invalid CBFs but require known obstacle dynamics. *Opportunity: Differentiable QP could enable gradient-based correction of CBF parameters when violations are detected.*

## Key References

The most important papers cited in the survey that are relevant to RoCBF-Net:

| Ref | Paper | Relevance |
|-----|-------|-----------|
| [5] | Ames et al., "Control Barrier Functions: Theory and Applications" (2019) | Foundational CBF-QP formulation |
| [11] | Cheng et al., "End-to-end Safe RL through Barrier Functions" (AAAI 2019) | CBF+GP safety filter for RL; probabilistic guarantees |
| [22] | Emam et al., "Safe RL Using Robust Control Barrier Functions" (2022) | Robust CBF + RL; closest to RoCBF-Net on uncertainty handling |
| [12] | Cheng et al., "Safe Model-free RL Using DOBCBF" (2022) | Disturbance observer CBF; high relative degree + uncertainty |
| [49] | Ma et al., "Model-based Constrained RL Using GCBF" (IROS 2021) | Generalized CBF for high relative degree + model-based RL |
| [50] | Ma et al., "Learning Differentiable Safety-Critical Control Using CBF" (2022) | **Closest prior**: differentiable CBF for generalization; ECBF without GP |
| [76] | Zhang et al., "Model-based RL with Provable Safety via CBF" (ICRA 2021) | UTCBF + GP; probabilistic guarantees in model-based RL |
| [58] | Robey et al., "Learning CBFs from Expert Demonstrations" (CDC 2020) | LfD for CBF construction |
| [44] | Lindemann et al., "Learning Hybrid CBFs from Data" (2020) | Safe-only demonstrations; constrained optimization |
| [45] | Lindemann et al., "Learning Robust Output CBFs" (2021) | ROCBF from demonstrations; model + estimation uncertainty |
| [15] | Dai et al., "Learning a Better CBF" (2022) | DDN-based CBF expansion from HCBF prior |
| [24] | Gao et al., "Online CBF for Decentralized Multi-Agent Navigation" (2023) | Adaptive $\alpha$ via GNN |
| [9] | Brunke et al., "Safe Learning in Robotics" (2021) | Broader safe RL survey; constraint categorization |
| [34] | Khan & Chatterjee, "Gaussian CBF: Safe Learning and Control" (CDC 2020) | GP-based CBF construction; online QP |
| [17] | Das et al., "Robust CBF with Uncertainty Estimation" (2023) | Robust CBF formalism with uncertainty |

## Relevance to RoCBF-Net

### Novelty Positioning
RoCBF-Net's novelty lies at the intersection of three streams that this survey treats separately:
1. **High relative degree CBFs** (ECBF [26], GCBF [49], DOBCBF [12]) -- none use differentiable QP
2. **GP-augmented CBF safety filters** ([11], [76], [34], [35]) -- none use HOCBF or differentiable QP
3. **Differentiable safety-critical control** ([50]) -- uses ECBF without GP robustness

No prior work combines all three. The survey's taxonomy (Table II, Table III) confirms this gap: no entry appears in both the "GP/uncertainty" row and the "differentiable/end-to-end" column.

### Related Work Section of Our Paper
When writing the related work section, the survey suggests organizing along:
1. **Safe RL with CBFs** (Section III-B2 of survey): Position RoCBF-Net as a CBF safety filter method with probabilistic guarantees. Cite [11], [22], [12], [76] as direct prior work on CBF+RL+uncertainty.
2. **High relative degree CBFs** (Section III-B2): Cite [49], [26], [12] as alternative approaches to HOCBF. Explain why HOCBF formulation is chosen (explicit derivative chain, compatibility with differentiable QP).
3. **Learning CBFs / CBF refinement** (Section IV): Distinguish RoCBF-Net from CBF construction methods -- we do not learn the CBF from data but rather enforce a known HOCBF robustly. Cite [15], [50], [24] as complementary work.
4. **Differentiable safety filters** (Section IV-A3): Cite [50] as the closest prior; explain how adding GP robustness to the differentiable QP framework is the key extension.

### Key Distinction
The survey repeatedly notes that "the ability to provide hard or probabilistic safety guarantees is contingent on having some form of system model" (Section V-A). RoCBF-Net's contribution is making this model requirement *adaptive* through GP, while keeping the safety filter *trainable* through differentiable QP -- bridging the gap between model-based guarantees and model-free flexibility.

## Cross-References

- [[xiao2019]] -- ECBF formulation for high relative degree (cited as [26] in survey context for Hailemichael et al. usage of ECBF)
- [[barriernet]] -- Neural network barrier certificate learning; related to the CBF refinement category
- [[ma2022]] -- Ma et al. (2022) "Learning Differentiable Safety-Critical Control Using CBF"; the closest prior work to RoCBF-Net's differentiable QP approach
- [[cheng2019]] -- Cheng et al. (AAAI 2019) "End-to-end Safe RL through Barrier Functions"; CBF+GP safety filter baseline
- [[emam2022]] -- Emam et al. (2022) "Safe RL Using Robust CBF"; robust CBF + RL
- [[zhang2021_utcbf]] -- Zhang et al. (ICRA 2021) UTCBF; model-based RL + GP + CBF
- [[dai2022]] -- Dai et al. (2022) "Learning a Better CBF"; DDN-based CBF expansion from priors
- [[robey2020]] -- Robey et al. (CDC 2020) "Learning CBFs from Expert Demonstrations"
- [[lindemann2021_rocbf]] -- Lindemann et al. (2021) "Learning Robust Output CBFs"; ROCBF from demonstrations
- [[khan2020_gcbf]] -- Khan & Chatterjee (CDC 2020) "Gaussian CBF"; GP-based CBF construction
- [[gao2023]] -- Gao et al. (2023) "Online CBF for Decentralized Multi-Agent Navigation"; adaptive $\alpha$
