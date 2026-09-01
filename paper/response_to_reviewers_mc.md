# Response to the Editor and Reviewers

**Manuscript ID:** MAC-26-0207
**Title:** *Commissioning-Calibrated GP-HOCBF Safety Filtering for Ultra-Supercritical Boiler-Turbine Control under Model Mismatch*

We thank the Editor and Reviewers for the detailed assessment. The revision substantially changes the validation and deployment presentation. The principal revisions are:

1. All primary simulations now satisfy the theorem's exact-input-matrix assumption, `Delta g = 0`; the former control-effectiveness-scaled results have been removed from the main evidence chain.
2. The distinction between the formal endpoint (`epsilon_kappa = 1`) and calibrated operating points (`epsilon_kappa < 1`) is stated in the abstract, theorem discussion, results, and conclusion.
3. Robustness-factor selection now uses a predeclared tune/test split. Seeds 0--2 select the smallest setting passing violation and QP-rejection gates; seeds 3--4 test the fixed setting without retuning.
4. A constrained NMPC reference, count-level QP feasibility diagnostics, and controlled GP data quantity/quality experiments have been added.
5. The GP commissioning procedure, fixed dictionary size, coverage monitoring, fallback state machine, deployment hardware, DCS integration, and computation bounds are now specified concretely.
6. Three additional high-load plant-controller windows extend execution evidence to 480.7--629.7 MW and distinguish routine full-QP operation from guarded reduced-QP recovery.
7. The figures, captions, notation, and English have been revised throughout. Old calibration and coupling-envelope material that no longer supports the revised evidence chain has been removed.

## Response to the Associate Editor

**Comment.** The reviews recognize the manuscript as a technically strong and practically relevant contribution. However, substantial issues remain regarding the distinction between theoretical safety guarantees and empirically calibrated margins. The validation methodology also requires stronger support, including clearer separation of empirical evidence from formal guarantees. In addition, the commissioning procedure, GP training and computational scalability, feasibility handling, and comparison with NMPC should be described more concretely.

**Response.** Theorem 1 is now explicitly limited to `epsilon_kappa = 1`, an independently valid simultaneous residual event, a valid compositional derivative bound, `Delta g = 0`, and full-row robust-QP feasibility. The revised text also distinguishes the implemented commissioning multiplier from an RKHS confidence multiplier containing the required residual-norm and noise-scale terms; the implemented multiplier does not establish the theorem assumptions by itself. Benchmark and plant values below one are described only as finite-sample commissioning operating points. The revised method section specifies the GP input/target semantics, frozen standardization, field data split, fixed 500-point dictionary, OOD monitors, and fallback state machine. The experimental section now includes NMPC in the primary table, a separate QP-rejection/fallback table, tune/test calibration, GP data-quality sensitivity, and high-load plant-controller evidence. The Supplemental Material provides count-level data, computational complexity, and controller-supervisor details.

**Location in the revised manuscript.** Sections 3.1--3.4; Sections 4.1--4.6; Tables 1--5; Figures 4, 5, and 8; Supplemental Sections S1--S6.

## Response to Reviewer 1

### Comment 1: English clarity and sentence length

**Comment.** The manuscript's English language should be reviewed and polished for clarity and fluency. Certain sections contain grammatical errors, awkward phrasing, and lengthy sentences that hinder comprehension. Paragraph transitions and complex sentences should be simplified for better readability.

**Response.** The manuscript has been edited throughout for sentence structure, transitions, terminology, and consistency. The Introduction and Related Work were shortened, the Experimental Validation section was reorganized by evidence type, and the Conclusion was rewritten to avoid repeating the abstract. Long captions were reduced, and implementation detail was moved to the Supplemental Material where appropriate.

**Location in the revised manuscript.** Sections 1 and 2; Section 4; Section 5; captions to Figures 2--8 and Tables 2--5.

### Comment 2: Figures 2--7 need clearer legends, axes, units, and symbols

**Comment.** Figures 2--7 are critical for demonstrating the method's performance but currently lack descriptive clarity. Improve the figure legends by clearly stating what each figure illustrates, specifying units, and explaining all axes and symbols.

**Response.** The central figures have been regenerated. Figure 4 now separates tune and held-out calibration data. Figure 2 uses native 1 s markers, identifies the interpolated display grid, gives physical margin units, and retains a full-rollout event strip. Figure 3 defines the residual-rate units, the implemented commissioning envelope, its multiplier, and the normalized residual ratio without presenting that plotted envelope as a separately proved uniform confidence set. Figure 5 defines training-set size, injected corruption, held-out NRMSE, interval coverage, violation rate, and QP-rejection rate. Figures 6 and 7 identify historian sampling intervals, physical units, matching metrics, and the interpretation boundary of the plant data. All captions identify the data source and statistical denominator where applicable.

**Location in the revised manuscript.** Figures 2--7 and their captions in Sections 4.2--4.5.

### Comment 3: Table 2 was referenced but missing

**Comment.** The referenced Table 2 appears in Section 4.1 but is not visible or included in the current version. Ensure that all tables are present, properly labelled, and self-explanatory.

**Response.** Table 2 is restored in the revised editable manuscript and is self-contained. It reports the nominal case and six mismatch conditions for the fixed proposal, no-GP HOCBF, constrained NMPC, the tune-selected RoCBF-SF setting, and the full implemented-margin endpoint. Because the predeclared selector chooses $\epsilon_\kappa=0$, the selected row is also the GP-mean-only ablation; we report it once rather than duplicate an identical numerical row. Table 3 separately reports count-level S3 QP rejection, fallback, and intervention.

**Location in the revised manuscript.** Section 4.2, Tables 2 and 3.

### Comment 4: Strengthen plant validation and clarify whether `epsilon_kappa` is fixed or adaptive

**Comment.** Additional quantitative metrics could strengthen validation on the 660 MW plant, including violation rates, false alarms, response times, and robustness under varying conditions. Clarify how `epsilon_kappa` is selected in practice and whether it is adapted during operation or fixed from commissioning data.

**Response.** The revised manuscript adds three high-load controller exports containing 4320 records at 5 s export spacing and covering 480.7--629.7 MW. It reports direct pressure, enthalpy, and power margins, QP status, reduced-QP recovery, saturation flags, QP timing, and total controller-task timing. We do not label QP interventions as false alarms because the exports do not contain an independent alarm-ground-truth label. The field value $\epsilon_\kappa=0.1$ is fixed after training-block replay and one-pass time-isolated validation; it is neither adapted online nor inferred from the benchmark tune/test selection, which independently selected the mean-only endpoint $\epsilon_\kappa=0$. The public controller exports confirm the implemented field value and execution outcome, while record-level plant replay remains a restricted enterprise asset.

**Location in the revised manuscript.** Sections 3.4, 4.4, and 4.5; Figures 6--8; Table 4; Supplemental Sections S3--S6.

### Comment 5: Sensitivity to GP data quantity and quality

**Comment.** How sensitive is the safety filter's performance to the quantity and quality of Gaussian-process residual data used for learning?

**Response.** A new controlled experiment varies the selected dictionary size (100, 250, and 500 transitions) and injects 0%, 5%, or 10% signed three-standard-deviation target corruption. Five seeds are evaluated for each of the nine size--corruption settings, giving 45 seeded fit-and-evaluation runs. Each run fits three scalar-output GPs, one for each component of `[p_m, h_m, N_e]`, for 135 scalar GP fits in total. Clean models have held-out NRMSE of 0.00046--0.00090, 100% empirical interval coverage, 0/2500 observed violations, and no QP rejection at each dictionary size. Under synthetic corruption, the 100- and 250-point dictionaries produce 5.76--20.24% violation and 4.04--8.60% rejection, whereas the 500-point dictionary retains 0/2500 observed violations and no rejection at both tested levels. We therefore use residual error, coverage, and closed-loop feasibility jointly rather than claiming that every corrupted fit fails. The corruption levels are explicitly identified as synthetic fault injection, not plant bad-point rates or a universal robustness bound.

**Location in the revised manuscript.** Section 4.3, Figure 5; Supplemental Section S3, Table S4.

### Comment 6: Systematic tuning of `epsilon_kappa`

**Comment.** How does `epsilon_kappa` affect feasibility and safety margins in practice? Is there a systematic method for tuning this parameter during plant commissioning?

**Response.** The revision replaces the previous same-data sweep with a tune/test procedure. Tune seeds 0--2 provide 15,000 samples per setting. The predeclared rule selects the smallest value for which pooled and maximum-per-seed violation rates are below 1% and QP-rejection rates are below 0.5%. Values from 0 to 0.05 all pass with zero observed violations and no rejected QP, so the smallest-passing rule selects $\epsilon_\kappa=0$. Held-out seeds 3--4 then give 0/50,000 observed violations and 0/50,000 QP rejections without retuning. Values of 0.2 and above fail the tune-set feasibility gate, and $\epsilon_\kappa=1$ rejects all 15,000 tune-set QPs. These are finite-sample commissioning outcomes, not a certificate or proof of zero event probability. The field value is calibrated independently by the same separation principle: training-block replay fixes $\epsilon_\kappa=0.1$, and the time-isolated validation block is evaluated once without retuning. Record-level field replay is access-restricted and is not reconstructed from the public controller exports.

**Location in the revised manuscript.** Sections 3.4, 4.2, 4.3, and 4.6; Figure 4; Tables 2, 3, and 5; Supplemental Section S2, Table S2.

### Comment 7: Unmodelled disturbances or faults outside training coverage

**Comment.** What are the limitations of the approach when the plant encounters unmodelled disturbances or faults that are not represented in the training data?

**Response.** The method section and Supplemental Material now define input z-score and quantile checks, posterior-variance checks, standardized one-step residual-rate innovation thresholds, and a bounded degraded mode. Moderate coverage loss withdraws GP mean correction and uses the full-row nominal HOCBF for at most ten cycles. Recovery requires ten consecutive healthy cycles. Persistent OOD or a hard fault transfers authority to the live DCS command and latches bypass. Severe innovation, invalid or stale measurements, communication loss, non-finite values, direct-bound failure, actuator infeasibility, or timeout bypasses the degraded stage immediately. The degraded mode is not described as certified.

**Location in the revised manuscript.** Section 3.4; Section 4.6, Table 5; Supplemental Section S4, Table S5.

### Comment 8: Scalability, real-time implementation, and unforeseen failures

**Comment.** Expand the limitations discussion to cover computational scalability, real-time constraints, and unforeseen plant failures, and discuss possible extensions such as adaptive margins or online residual learning.

**Response.** The deployed GP dictionary is frozen at 500 points; online sample growth is zero. Exact-GP storage and retraining complexity are stated as `O(qN^2)` and `O(qN^3)` for three outputs, with approximately 3 MB for float32 Cholesky factors and 10--20 MB for model arrays. The controller hardware, operating system, Python and solver versions, Kubernetes management, and Modbus TCP interface are reported. High-load exports show a maximum QP time of 6.234 ms and maximum total task time of 28.665 ms against a 1000 ms deadline. Coverage loss triggers degraded operation or offline recalibration rather than unrestricted online GP growth or online margin escalation. Future work addresses actuator-gain uncertainty and sampled-data certification.

**Location in the revised manuscript.** Sections 3.4, 4.4, and 4.5; Figure 8; Table 4; Section 5; Supplemental Sections S3--S6.

## Response to Reviewer 2

### Comment 1: GP complexity growth over long operation

**Comment.** Online GP inference can suffer from computational-complexity growth as the dataset expands. Clarify how training sample sizes, memory gating, or sparse-GP techniques are managed over long operational horizons to maintain bounded solve time.

**Response.** The field GP is trained offline and frozen. It uses three exact Matérn-5/2 GPs with 500 selected transitions and 200 time-isolated validation transitions. The model is reviewed quarterly and after major maintenance or a material coal-source change, but the online dictionary does not grow. The manuscript gives the data-selection procedure, memory estimates, asymptotic complexity, and rollback policy. Sparse GP methods are identified as an extension if a larger dictionary becomes necessary; they are not claimed to be used in the current deployment.

**Location in the revised manuscript.** Sections 3.1 and 3.4; Supplemental Section S3.

### Comment 2: Transferable calibration guideline

**Comment.** Provide a brief guideline or rule of thumb to help engineers calibrate `epsilon_kappa` for other multi-input, actuator-bounded industrial processes.

**Response.** The revised guideline is procedural rather than a universal numerical value: define violation and QP-rejection gates; sweep candidate margins only on the training/replay block; select the smallest value passing pooled and worst-seed gates; test it once on time- or seed-isolated data; and enable it only if GP coverage and QP feasibility also pass. During operation, coverage loss triggers degraded mode or recalibration rather than online escalation of the margin. The benchmark selected $\epsilon_\kappa=0$, whereas the independently calibrated field value is 0.1. The difference is retained because it demonstrates that the rule can select either a mean-only or positive-margin operating point from the local residual and feasibility evidence. The field replay passed the time-isolated validation procedure, while its record-level data remain an access-restricted enterprise asset.

**Location in the revised manuscript.** Sections 3.4, 4.3, and 4.6; Figure 4; Table 5; Supplemental Sections S2 and S4.

### Comment 3: Exact fallback mechanism

**Comment.** Add a note detailing the exact fallback mechanism or secondary safety logic used if the QP solver fails to return a feasible solution during an unexpected edge-case disturbance.

**Response.** The revised manuscript specifies one guarded reduced-QP retry for the whitelisted `pressure_low` row only. It requires normalized row authority below 0.01, raw RHS below `-1e-6`, direct pressure margin of at least 2.0 MPa, valid measurements, and no active protection or interlock. Actuator box/rate rows are never removed. A failed retry, timeout, non-finite result, or hard fault uses the current live DCS command and can latch bypass. The deployment configuration specifies re-entry after healthy measurements, heartbeats, dry-run QPs, operator acknowledgement for latched faults, Kubernetes Lease ownership, and bumpless transfer. Controller exports verify execution status, timing, and recovered-QP records, but do not expose every heartbeat or lease transition.

**Location in the revised manuscript.** Sections 3.1 and 3.4; Section 4.5, Figure 8 and Table 4; Supplemental Sections S4 and S5.

## Response to Reviewer 3

### Comment 1: Finite zero-violation results are not probabilistic guarantees

**Comment.** The abstract's zero-violation result is finite-sample evidence rather than a statistical guarantee. The Wilson bound is not sufficiently contextualized, and empirical calibration of `epsilon_kappa` may invalidate the formal certificate. Separate empirical validation from theoretical guarantees, state that the calibrated value has no formal probabilistic guarantee, and present zero violations only as observed in the tested rollouts.

**Response.** The abstract, results, and conclusion now use count-level language and explicitly distinguish the primary sweep from the held-out test. The predeclared S3 tune/test procedure selects the smallest passing value, `epsilon_kappa = 0`. With this fixed mean-only setting, RoCBF-SF records 0/175,000 observed violations across the nominal case and six mismatch conditions in the primary sweep. Each condition retains its own frozen scenario-specific GP; the controller does not identify the true scenario online. Held-out S3 seeds then record 0/50,000 violations and 0/50,000 QP rejections without retuning. Wilson limits are described only as descriptive Bernoulli-reference values because controller samples are serially correlated. No zero count is presented as proof of zero event probability or as a formal probabilistic guarantee.

**Location in the revised manuscript.** Abstract; Sections 4.1--4.3; Tables 2 and 3; Figure 4; Section 5; Supplemental Sections S1 and S2.

### Comment 2: The theorem applies at `epsilon_kappa = 1`, whereas the deployed value is calibrated

**Comment.** Clarify whether the deployed `epsilon_kappa = 0.1` system is certified or calibrated. If it is calibrated, the paper should not invoke Theorem 1 for that deployed operating point. Consider distinguishing a certified full-margin mode from a calibrated partial-margin mode with different safety arguments.

**Response.** This distinction is now explicit throughout. Theorem 1 applies at `epsilon_kappa = 1` only when the simultaneous residual event, compositional derivative bounds, `Delta g = 0`, and full-row QP feasibility are independently established. The numerical `epsilon_kappa = 1` row is therefore called the full implemented-margin endpoint rather than being treated as automatically certified. The benchmark tune/test procedure selects the mean-only endpoint `epsilon_kappa = 0`, and the field uses an independently calibrated value of 0.1; both are empirical commissioning operating points rather than certified settings. The method section gives the standard RKHS-form multiplier with explicit residual-norm and noise-scale terms and separately identifies the fixed multiplier used in experiments and deployment as an engineering commissioning rule. The theorem is not invoked for the field value, the finite-rollout calibration results, or guarded reduced-QP cycles.

**Location in the revised manuscript.** Section 3.4; Table 2 caption; Sections 4.2, 4.3, 4.5, and 4.6; Section 5.

### Comment 3: Before/after limitations and insufficient load coverage

**Comment.** The before/after design does not by itself isolate the effect of the proposed controller. The load match is imperfect, combustion conditions differ, only low-to-mid-load windows were presented, and maintenance or cleaning during the outage could also explain improvement. Acknowledge these limitations, provide evidence about what changed, validate beyond 200--350 MW, explain the increased air--coal-ratio dispersion, and consider alternative retrofit explanations.

**Response.** The before/after comparison is now described as observational evidence associated with the documented retrofit, not a standalone causal estimate. The outage included instrument calibration, actuator inspection, combustion-equipment maintenance, and heat-surface cleaning. At the same time, the original PID gains, limits, anti-windup, sliding-pressure curve, coordinated-control logic, and base fuel/feedwater/turbine-valve loops were unchanged; RoCBF-SF was added downstream. The increased air--coal-ratio dispersion is retained and discussed as evidence that the post-retrofit pair was not uniformly easier. We also rebuilt the historian cohort from the source query files: 14,675 eligible pre-retrofit and 517 post-retrofit windows yield 512 one-to-one matches without replacement. The median pressure-error standard deviation changes from 0.523 to 0.502 MPa, but the post-day cluster-bootstrap interval crosses zero; this cohort is therefore used as operating-context consistency evidence rather than as a precise causal effect estimate. The separate native-5 s pair supplies the high-resolution response diagnostic.

Three new confirmed controller exports extend execution evidence to 480.7--629.7 MW (72.8--95.4% of nameplate). The original low-to-mid-load records are retained only as historian operating context and are not used to claim controller-internal execution. The new exports establish high-load controller execution, but they do not constitute a high-load pre/post causal-performance comparison, and the revision does not claim response improvement at every load. The matched-historian response claim and the controller-execution claim are kept separate.

**Location in the revised manuscript.** Section 4.5, Figures 6--8 and Table 4; Section 5; Supplemental Section S5.

### Comment 4: The previous control-effectiveness scaling violated `Delta g = 0`

**Comment.** Theorem 1 assumes an exact input matrix (`Delta g = 0`), but the previous stress tests used pressure-scaled control effectiveness. Either extend the theory to `Delta g != 0` or state clearly that such tests are outside the certificate.

**Response.** The primary validation has been rerun with a fixed input matrix for both the plant and filter, and all mismatch enters through drift residuals. The former pressure-scaled control-effectiveness table, captions, and associated main-text conclusions have been removed. S5 is implemented as a drift-equivalent process perturbation in the $\Delta g=0$ benchmark. This aligns the experiment with the theorem's input-matrix assumption, while the revised manuscript avoids claiming that finite rollouts verify its separate uniform residual and derivative-bound assumptions. The paper does not claim an actuator-gain-uncertainty certificate; this extension is listed as future work.

**Location in the revised manuscript.** Sections 3.2 and 3.4; Sections 4.1 and 4.2, Table 2; Section 5.

### Comment 5: The commissioning protocol was not operational

**Comment.** Define the local replay screen and supervised commissioning data quantitatively, explain how operating regimes are detected, address time-varying mismatch, and specify a monitoring scheme that triggers recalibration.

**Response.** The former gamma-based regime table has been removed because it required an unimplemented online estimate of gamma. The replacement specifies a 14-day field-data split, a 24 h isolation gap, five fixed load strata, 60 s thinning, deterministic farthest-point selection, no quota borrowing, tune/test margin gates, OOD thresholds, innovation thresholds, recovery counts, timeouts, and recalibration triggers. Time-varying mismatch is detected from coverage and innovation statistics rather than an unobservable coupling label.

**Location in the revised manuscript.** Section 3.4; Section 4.6, Table 5; Supplemental Sections S3 and S4, Tables S3 and S5.

### Comment 6: Novelty relative to robust MPC and simpler alternatives

**Comment.** Excessive constraint tightening and infeasibility are already known in robust control. Clarify the specific non-trivial contribution for boiler--turbine control, compare against GP mean correction without uncertainty propagation, and explain the distinction from existing GP-HOCBF methods.

**Response.** The revised text no longer claims novelty for the generic observation that excessive tightening can cause infeasibility. The contribution is stated more narrowly: an overlay safety filter for command-level relative-degree-two boiler--turbine constraints obtained from an actuator-augmented model, GP residual-rate correction in `[p_m,h_m,N_e]`, uncertainty propagation through the HOCBF chain, and a plant-oriented commissioning/fallback procedure. The final primary table and numerical comparison are regenerated from the revised seven-state benchmark rather than carried over from the earlier five-state implementation.

**Location in the revised manuscript.** Sections 1.3 and 1.4; Sections 2.2--2.4; Sections 3.2 and 3.3; Section 4.2, Table 2.

### Comment 7: Deployment hardware, DCS integration, and CPU timing

**Comment.** Specify the plant hardware, discuss DCS integration, memory and reliability constraints, clarify the interpretation of the previous 25 ms timing, and provide CPU-based timing evidence.

**Response.** The deployment configuration uses an AMD EPYC Embedded 4565P processor, 64 GB DDR5-5600 ECC memory, openEuler 24.03 LTS, Python 3.11, qpax 0.1.3, SciPy 1.17.1, Kubernetes, and Modbus TCP. The controller period and task deadline are 1000 ms. Confirmed high-load controller exports report maximum QP and total-task times of 6.234 and 28.665 ms on the deployed CPU. The previous GPU research-pipeline timing is no longer used as field deployment evidence. The configuration specifies Kubernetes Lease ownership and DCS-side heartbeat/sequence gating, with the DCS retaining final write authority; the exports do not expose individual fencing transitions.

**Location in the revised manuscript.** Section 3.1, Table 1; Sections 4.4 and 4.5, Figure 8 and Table 4; Supplemental Sections S3, S5, and S6.

### Comment 8: Notation, equation numbering, and the former Figure 10

**Comment.** Clarify the identity/one/ell notation, define the deviation-input set, correct equation numbering, and split the dense former Figure 10 or provide a guided interpretation.

**Response.** The physical command `u`, deviation command `v`, proposed command `v_prop`, accepted command `v*`, and deviation-input box `V` are now defined together. The revision also reserves `A_0,b_0` for the no-GP nominal ablation, uses `A,b` consistently for the GP-mean-corrected online rows, and explicitly defines the recursive HOCBF admissible set in Theorem 1. Identity/one/ell ambiguities and equation references were checked, and the compositional margin is presented as a numbered equation block. The former dense plant presentation is now separated into Figure 7 for the matched historian response and Figure 8 for high-load controller execution and recovery, with a guided interpretation in the text.

**Location in the revised manuscript.** Sections 3.1--3.3; Section 4.5, Figures 7 and 8.

### Comment 9: NMPC was not systematically compared

**Comment.** Include NMPC in the main results, avoid unsupported claims of superiority over MPC, and acknowledge that a well-tuned constrained MPC may achieve similar or better safety.

**Response.** A constrained NMPC reference is now included in the primary main-text table for the nominal case and all six mismatch scenarios. Its horizon, weights, bounds, constraints, warm start, additive disturbance correction, SLSQP tolerance, convergence check, and timing are reported. It records 0/175,000 observed violations and no solver failure, matching the tune-selected RoCBF-SF row on the reported violation count. The revision therefore does not claim that RoCBF-SF is universally safer than MPC. The overlay's engineering distinction is compatibility with an existing coordinated controller and a one-step downstream projection.

**Location in the revised manuscript.** Sections 2.2, 4.1, and 4.2; Table 2; Supplemental Section S6.

### Comment 10: Actuator limits and feasibility implications

**Comment.** The manuscript treats actuator limits as box constraints but does not sufficiently analyse their implications for QP feasibility.

**Response.** The revised QP diagnostics record every attempt, rejection, fallback, and intervention. In S3, the full implemented-margin endpoint rejects 25,000/25,000 QPs, reverts to the upstream proposal, and records 9798/25,000 violating samples. Across all six mismatch conditions it rejects 149,800/150,000 QPs. These counts directly connect excessive implemented tightening to actuator-limited infeasibility in the tested benchmark. The field section separately reports the guarded pressure-low-row recovery and explicitly excludes recovered cycles from the full-row certificate. Actuator box and rate rows are never removable.

**Location in the revised manuscript.** Sections 3.1 and 3.4; Section 4.2, Table 3; Section 4.5, Figure 8 and Table 4; Supplemental Sections S1, S4, and S5.

### Comment 11: Training-data collection, coverage, and online updating

**Comment.** Clarify how training data were collected, whether the benchmark scenarios use separate training data, what happens outside training coverage, and whether the GP is updated online.

**Response.** Benchmark GPs use scenario-specific training transitions only for controlled mechanism isolation. The field GP uses approximately 14 days of 1 Hz commissioning data, with days 1--10 for candidates, day 11 as an isolation gap, and days 12--14 for validation. Five load strata contribute 100 training and 40 validation points each after filtering and 60 s thinning. The model is frozen online; coverage or innovation failure withdraws GP correction and triggers degraded operation, bypass, or offline recalibration.

**Location in the revised manuscript.** Sections 3.2 and 3.4; Section 4.1; Supplemental Sections S3 and S4.

### Comment 12: Justification of the 1 s loop

**Comment.** The control loop runs at 1 s sampling, whereas the dominant plant dynamics have time constants of seconds to minutes.

**Response.** One second is the actual deployed coordinated-control task period, not a time constant assigned to the boiler thermal dynamics. The filter evaluates each upstream command update before the command-selection stage. The controller exports are decimated to 5 s records, so their spacing is a logging interval rather than the execution period. The measured total-task maximum is 28.665 ms under a 1000 ms deadline. We also state the theoretical boundary explicitly: Theorem 1 is continuous-time and is not claimed as a sampled-data certificate for the 1 s zero-order-hold implementation. A sampled-data GP-HOCBF certificate remains future work.

**Location in the revised manuscript.** Sections 3.1 and 3.4; Sections 4.4 and 4.5; Section 5; Supplemental Section S6.
