# ferx-core

[![CI](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/ci.yml)
[![Slow tests](https://github.com/FeRx-NLME/ferx-core/actions/workflows/slow-tests.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/slow-tests.yml)
[![Docs](https://github.com/FeRx-NLME/ferx-core/actions/workflows/docs.yml/badge.svg)](https://github.com/FeRx-NLME/ferx-core/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/FeRx-NLME/ferx-core/branch/main/graph/badge.svg)](https://codecov.io/gh/FeRx-NLME/ferx-core)
[![CodeFactor](https://www.codefactor.io/repository/github/ferx-nlme/ferx-core/badge)](https://www.codefactor.io/repository/github/ferx-nlme/ferx-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics, written in Rust. Implements FOCEI and SAEM estimation with analytical PK solutions and ODE solvers.

Additional features:
- PK-PD and multi-analyte modeling
- BLQ likelihood modeling
- Importance Sampling & SIR
- Deep Compartmental Models & Neural ODEs
- Stochastic differential equations
- Simulation with uncertainty
- Various optimizers
- ... and more

## Quick Start

```bash
# Build
cargo build --release

# Fit a model
cargo run --release --bin ferx -- examples/warfarin.ferx --data data/warfarin.csv

# Fit with simulated data (uses [simulation] block)
cargo run --release --bin ferx -- examples/warfarin.ferx --simulate
```

Output files: `{model}-fit.yaml` (parameter estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Model File Format (.ferx)

Models are defined in a simple DSL. Here is a one-compartment oral PK model for warfarin:

```
[parameters]
  theta TVCL(0.2, 0.001, 10.0)     # name(initial, lower, upper)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09              # between-subject variability (variance)
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02            # residual error

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

## Structural Models

| Model | Syntax |
|-------|--------|
| 1-compartment IV (bolus and/or infusion) | `pk one_cpt_iv(cl=CL, v=V)` |
| 1-compartment oral | `pk one_cpt_oral(cl=CL, v=V, ka=KA)` |
| 2-compartment IV (bolus and/or infusion) | `pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)` |
| 2-compartment oral | `pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)` |
| 3-compartment IV (bolus and/or infusion) | `pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)` |
| 3-compartment oral | `pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)` |
| ODE-based | Define equations in an `[odes]` block |

For IV models, the closed form (bolus vs infusion) is chosen per dose event from the `RATE` column — a subject can mix bolus and infusion records.

## Estimation Methods

Set via `method` in `[fit_options]`:

| Method | Description |
|--------|-------------|
| `foce` | First-Order Conditional Estimation |
| `focei` | FOCE with Interaction (default) |
| `gn` | Gauss-Newton (BHHH) with Levenberg-Marquardt damping |
| `gn_hybrid` | Gauss-Newton followed by FOCEI polish |
| `saem` | Stochastic Approximation EM |
| `imp` | Importance Sampling (typically chained after another method for OFV evaluation) |

Methods can be chained (e.g. `method = saem, focei, imp`) to run sequentially.

### Optimizers

For FOCE/FOCEI, the outer optimizer can be set via `optimizer` in `[fit_options]`:

| Optimizer | Description |
|-----------|-------------|
| `slsqp` | NLopt Sequential Least Squares Programming (default) |
| `lbfgs` | NLopt L-BFGS |
| `mma` | NLopt Method of Moving Asymptotes |
| `bfgs` | Built-in BFGS |
| `bobyqa` | NLopt BOBYQA (derivative-free) |
| `trust_region` | Newton trust-region (argmin + Steihaug CG) |

## Vine Copula Between-Subject Variability (`omega_dist = vine`)

### What it is and why it matters

Every NLME model ever fitted in NONMEM, nlmixr2, or the standard ferx FOCE/SAEM path makes the same assumption: the between-subject random effects η follow a **multivariate normal** distribution, η ~ N(0, Ω). The OMEGA matrix fully describes the shape of that distribution — its variances and covariances.

This assumption is convenient but restrictive. Real patient populations sometimes show:

- **Heavy tails** — more extreme subjects than a Gaussian predicts, inflating individual EBE-based residuals
- **Skewed marginals** — clearance distributions with a longer upper tail than lower (or vice versa)
- **Tail-dependent correlations** — subjects at the extremes of CL and V may cluster together more (or less) strongly than subjects near the centre of the distribution. A Pearson correlation in Ω cannot distinguish between "highly correlated only in the tails" and "uniformly correlated everywhere"

These features can cause:
- Biased population parameter estimates (θ, Ω) when the Gaussian misspecification is severe
- Inflated residuals and poor CWRES/IWRES diagnostics in specific subject groups
- Suboptimal individual predictions (IPREDs) for outlier subjects

`omega_dist = vine` replaces the multivariate-Gaussian BSV with a **D-vine pair-copula construction**. The marginal distribution of each η_i remains Gaussian (so the OMEGA diagonal is still interpretable), but the joint dependence structure is estimated from the data and can be non-Gaussian. Estimation runs via SAEM with a vine-prior Metropolis-Hastings E-step.

---

### For pharmacometricians

**The intuition**

In a standard fit, OMEGA simultaneously describes two things: how spread out each ETA is, and how the ETAs are related to each other. The vine approach separates these:

1. **How spread out is each ETA?** — still described by the OMEGA diagonal and reported as the marginal SD. Nothing changes here.
2. **How are the ETAs related to each other?** — instead of a single Pearson correlation, a bivariate *copula* is fitted for each ETA pair. Different copula families can capture different shapes of dependence.

The copula families available are:

| Family | What it captures |
|--------|-----------------|
| `gaussian` | The usual symmetric correlation (same as standard OMEGA off-diagonal) |
| `student_t` | Symmetric correlation but with heavier joint tails — extreme subjects in *both* ETAs simultaneously are more likely than a Gaussian predicts |
| `clayton` | **Lower** tail dependence — patients with unusually *low* values in both parameters tend to cluster together |
| `gumbel` | **Upper** tail dependence — patients with unusually *high* values in both parameters tend to cluster together |
| `frank` | Symmetric dependence without tail effects, more flexible than Gaussian |

Families are selected automatically by AIC on the first SAEM M-step and frozen for the rest of the run.

**When should you try this?**

- When η histograms or κ–η scatterplots look non-normal or show curvature at the extremes
- When you observe a subset of subjects with consistently poor IPREDs despite adequate average fit
- As a diagnostic: if every pair-copula selects `gaussian`, your standard model BSV structure is confirmed adequate
- When comparing competing structural models and you want the fairest possible OFV baseline

**Quick start**

Add `omega_dist = vine` to any model that already has a `method = saem` line:

```
[fit_options]
  method     = saem
  omega_dist = vine
```

**Note on method chaining**: `omega_dist = vine` requires `method = saem`. Chaining with `focei` (e.g. `method = saem, focei`) is rejected at parse time — FOCEI's inner objective uses the Gaussian quadratic prior, which is incompatible with the vine copula prior. For final inference use the SAEM result directly; the vine-corrected OFV is the appropriate quantity for model comparison.

**Reading the output**

The vine section in the console and YAML output reports, for each ETA pair at each tree level:

- **Family** — which copula was selected
- **Parameter** with approximate SE — e.g. `rho=0.431 (SE≈0.087)` for a Gaussian or Student-t pair, `theta=1.82 (SE≈0.41)` for Clayton/Gumbel/Frank
- **Kendall τ** — rank-based correlation, comparable across all families (unlike Pearson r which only makes sense for Gaussian)
- **λL / λU** — lower and upper tail dependence coefficients; values above ~0.1 indicate meaningful tail clustering

The `ΔOFV (Gaussian − corrected)` value in the output is the most important summary: a positive value means the vine model fits the data better than a Gaussian BSV on the same structural model, with full accounting for the extra copula parameters in the OFV baseline. Rule of thumb: ΔOFV > 3.84 per additional copula parameter (chi-squared approximation) suggests a meaningful improvement.

> **Note on the two OFV values reported:** The standard FOCE OFV uses the Gaussian-equivalent OMEGA and is printed for consistency with Gaussian fits. The *vine-corrected OFV* replaces the Gaussian BSV prior with the vine prior at the final EBEs — this is the value used for AIC and BIC and is the fair comparator to a Gaussian FOCEI OFV.

**IOV models**

Vine fits fully support models with inter-occasion variability (`kappa` declarations). The vine applies to the BSV (η) layer only. The IOV (κ) layer retains a Gaussian Ω_iov. This is scientifically appropriate: the vine captures cross-subject structure in the BSV distribution, while IOV captures within-subject occasion-to-occasion variability — a lower-signal process for which a vine copula would not be identifiable from typical NLME datasets.

---

### For statisticians

**Model**

The joint BSV density is factored as a product of Gaussian marginals and a D-vine copula density:

```
p(η₁, …, η_d) = ∏ᵢ φ(ηᵢ; μᵢ, σᵢ)  ×  c_vine(Φ(z₁), …, Φ(z_d))
```

where zᵢ = (ηᵢ − μᵢ)/σᵢ and uᵢ = Φ(zᵢ) is the probability integral transform (PIT). The D-vine copula density decomposes over d−1 tree levels into bivariate pair-copula densities via the sequential h-function recursion (Aas et al. 2009):

```
c(u₁,…,u_d) = ∏_{k=1}^{d-1} ∏_{j=0}^{d-k-1} c_{j,j+k|j+1,…,j+k-1}(u_{j|·}, u_{j+k|·})
```

The D-vine ordering is fixed at 0, 1, …, d−1 (no permutation selection in the current implementation).

**Estimation**

SAEM with a vine-prior Metropolis-Hastings E-step. The complete-data log-likelihood uses:

```
log p(ηᵢ) = Σⱼ [−½zᵢⱼ² − ½log(2π) − log σⱼ] + log c_vine(u₁,…,u_d)
```

The M-step separates into:
- **Marginals** (MLE, closed form): μⱼ ← mean(ηⱼ), σⱼ ← std(ηⱼ) from current SAEM samples
- **Pair-copulas** (IFM — Inference Functions for Margins): each pair copula is independently refitted by maximising the pair log-likelihood at the conditional pseudo-observations obtained by sequential h-function application through the vine tree structure. Family selection (AIC over the five candidate families above) occurs at the first M-step and is frozen thereafter to prevent log-prior discontinuities from disturbing the chain.

**Vine-corrected OFV**

The `vine_corrected_ofv` replaces the Gaussian prior in the standard FOCE population NLL with the vine prior at the post-SAEM final EBEs:

```
OFV_vine = OFV_FOCE  +  2 × Σᵢ [ log p_vine(η̂ᵢ) − log p_Gauss(η̂ᵢ; Ω_equiv) ]
```

This correction removes the Laplace-approximation error from the prior mismatch and yields an OFV that is directly comparable to a Gaussian FOCEI OFV on the same data.

**AIC / BIC**

```
k  =  k_theta  +  k_Ω (free Cholesky elements)  +  k_sigma  +  Σ_{pairs} n_params(family)
AIC = OFV_vine + 2k
BIC = OFV_vine + k × ln(N_obs)
```

Student-t pair-copulas contribute 2 (ρ and ν); all other families contribute 1.

**Copula parameter standard errors**

Per-pair SEs are approximate under the IFM assumption (pseudo-observations treated as fixed). The observed-information Hessian is computed by finite differences on the transformed scale:

| Family | Transform ψ(θ) | Back-transform SE |
|--------|---------------|-------------------|
| Gaussian / Student-t ρ | arctanh(ρ) | SE_ρ ≈ (1 − ρ²) × SE_ψ |
| Student-t ν | ln(ν) | SE_ν ≈ ν × SE_ψ |
| Clayton θ | ln(θ) | SE_θ ≈ θ × SE_ψ |
| Gumbel θ | ln(θ − 1) | SE_θ ≈ (θ − 1) × SE_ψ |
| Frank θ | identity | SE_θ = SE_ψ |

Rigorous SEs require a Godambe sandwich estimator (not yet implemented) that accounts for the uncertainty in the marginal PIT.

**Simulation**

Draws from the fitted vine use the inverse Rosenblatt transform (exact, O(d²) per draw): d independent U(0,1) variates are mapped through sequential h-inverse functions from the outermost tree level inward, then inverted through the Gaussian marginals. No MCMC is required.

---

### Fit options reference

| Option | Values | Default | Notes |
|--------|--------|---------|-------|
| `omega_dist` | `gaussian`, `vine`, `vine-multimodal` | `gaussian` | Must also set `method = saem` |
| `saem_n_exploration` | integer | 200 | Iterations for vine family selection and chain warm-up |
| `saem_n_convergence` | integer | 100 | Iterations for parameter convergence |
| `saem_omega_burnin` | integer | 50 | Iterations before vine M-step begins (chain warm-up) |
| `saem_n_mh_steps` | integer | 3 | MH proposals per subject per SAEM iteration |
| `covariance` | `true`/`false` | `false` | Covariance step for θ/σ SEs; copula SEs always computed |

---

### Output reference (vine additions)

Console output adds a `--- Vine Copula ---` block after the standard OMEGA/SIGMA table. YAML output adds a `vine_copula:` top-level key. Example (3 ETAs, tree 1 has 2 pairs, tree 2 has 1):

```yaml
vine_copula:
  marginals:
    ETA_CL:  { mean: -0.008, sd: 0.302 }
    ETA_V:   { mean:  0.003, sd: 0.194 }
    ETA_KA:  { mean:  0.011, sd: 0.556 }
  trees:
    - tree: 1                              # unconditional pairs
      pairs:
        - label: "ETA_CL ~ ETA_V"
          family: student_t
          rho:    0.431
          rho_se: 0.087                    # approximate SE (IFM)
          nu:     8.2
          nu_se:  3.1
          kendall_tau: 0.283
          tail_dep_lower: 0.124
          tail_dep_upper: 0.124
        - label: "ETA_V ~ ETA_KA"
          family: clayton
          theta:    1.24
          theta_se: 0.38
          kendall_tau: 0.383
          tail_dep_lower: 0.402
    - tree: 2                              # conditioned on ETA_V
      pairs:
        - label: "ETA_CL ~ ETA_KA | ETA_V"
          family: gaussian
          rho:    0.104
          rho_se: 0.096
          kendall_tau: 0.066

  ofv_vine_corrected:       1243.218       # use this for model comparison
  delta_ofv_vine_advantage:   12.441       # positive = vine improves fit vs Gaussian BSV
```

All standard output (theta, omega, sigma, AIC, BIC, sdtab, EBEs) is unchanged. AIC and BIC are computed from `ofv_vine_corrected` with the augmented parameter count.

---

### Limitations and roadmap

- **Variable ordering**: the D-vine ordering is currently fixed as declared in `[parameters]`. Optimal ordering or R-vine structure selection is not yet implemented.
- **Copula SEs**: approximate only (IFM assumption). The IFM pseudo-observations are treated as fixed, so the SEs do not propagate uncertainty from the marginal estimation step. A Godambe sandwich correction would be more rigorous but is not currently implemented.
- **HMC E-step**: the vine path always uses Metropolis-Hastings, regardless of `saem_n_leapfrog`. The gradient of the vine log-prior is not yet implemented, so HMC is silently ignored for vine fits (a warning is emitted). HMC (via the `autodiff` feature) is only available for Gaussian SAEM with analytical PK models.
- **EBEs under Gaussian prior**: final EBEs are optimised using the Gaussian-equivalent OMEGA, not the vine prior. CWRES, IWRES, and NPDE are therefore computed at Gaussian-optimal EBEs. For ETAs with strong non-Gaussian dependence, this is a known first-order approximation shared with all vine-based SAEM methods.
- **FOCEI chain**: `method = saem, focei` is rejected at parse time — FOCEI's Gaussian prior is incompatible with the vine prior.
- **ODE structural models**: fully supported — `[odes]`-based models work with `omega_dist = vine`. The MH E-step evaluates the ODE solver inside each proposal exactly as in Gaussian SAEM.

---

### References

- Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics* **44**, 182–198.
- Joe, H. (1996). Families of m-variate distributions with given margins and m(m−1)/2 bivariate dependence parameters. *IMS Lecture Notes–Monograph Series* **28**, 120–141.
- Delattre, M., Lavielle, M., Poursat, M.-A. (2014). A note on BIC in mixed-effects models. *Electronic Journal of Statistics* **8**(1), 456–475.

---

## Vine Copula with Mixture Marginals (`omega_dist = vine-multimodal`)

### Audit fixes (June 2026)

Following a three-way audit (statistician, Rust engineer, pharmacometrician), the following issues were corrected:

| # | Fix |
|---|-----|
| 1 | **Simulation wired**: `simulate()` / VPC now draws from the fitted mixture-vine distribution via inverse-Rosenblatt. Previously it fell through to a Gaussian draw, making VPCs silently wrong. |
| 2 | **FOCEI chain documented**: README no longer recommends `method = saem, focei` — the code rejects this combination at parse time (FOCEI uses a Gaussian prior incompatible with the mixture prior). |
| 3 | **Limitations updated**: "Number of components fixed at 2" was incorrect; k ∈ {1,…,4} is implemented. The Limitations section now accurately describes what is and isn't supported. |
| 4 | **AIC/BIC formula corrected**: formula now reads `3(k−1)×d`, not `3×d` (the old formula was only correct for k=2). |
| 5 | **Non-SAEM guard**: `omega_dist = vine / vine-multimodal` with `method = foce` or `method = gn` now returns a clear `E_OMEGA_DIST_NO_SAEM` error instead of silently discarding the vine settings and running a plain FOCE/GN fit. |
| 6 | **Log-sum-exp in E-step**: EM responsibilities are now computed in log-space with a stable log-sum-exp, preventing silent float64 underflow when ETA samples are far from all component means. `log_pdf` was also fixed to use the same approach. |
| 7 | **BIC k-selection pooled**: k selection by BIC now uses all η samples accumulated throughout the burn-in phase (burn-in iterations × N subjects), not just a single N-point snapshot. Dramatically improves statistical power for typical PK datasets (N=50–150). |
| 8 | **AIC/BIC mean note**: Added documentation note that the overall mixture mean is not explicitly pinned to zero, so the AIC/BIC penalty may underestimate by 1 per free-mean ETA in models without mu-referencing. |
| 9 | **Per-subject membership probabilities**: The sdtab now includes `COMP_<ETA>` (most probable 1-based component, per observation row) and `PROB<j>_<ETA>` (posterior probability of each component) for each ETA with k ≥ 2. |
| 10 | **No-panic `new()` / `fit_em()`**: `assert!` replaced with `debug_assert!` + silent clamp in release builds; k is clamped to [1, MAX_MIXTURE_COMPONENTS] rather than panicking on invalid input. |

### Follow-up corrections (June 2026)

Three additional numerical correctness bugs were found during demo-dataset smoke testing and
verified algebraically before fixing. Each had a measurable impact on reported results.

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 11 | **Vine-corrected OFV inflation** — The vine-corrected OFV (reported in `*-fit.yaml` as `ofv_vine_corrected`) was computed by adding the mixture log-prior *in its fully-normalized form*, while the Gaussian prior is computed in the FOCE convention (dropping the ½·d·log(2π) constant). The mismatch inflated the corrected OFV by exactly N·d·½·log(2π) — for a typical 80-subject, 2-ETA model this is ≈ 146 OFV units, making the vine model appear ~146 worse than it truly is. With the fix the ΔOFV vs. Gaussian correctly reflects only the genuine fit improvement from the non-Gaussian prior. | ΔOFV artefact of +N·d·½·log(2π) ≈ 146 for N=80, d=2. Caused vine AIC/BIC to systematically appear worse than Gaussian despite being a better fit. | Strip the ½·d·log(2π) term from the mixture prior NLL before computing the vine-corrected OFV delta, matching the FOCE convention used everywhere else. |
| 12 | **BIC k-selection used pooled sample count as N** — Fix #7 (above) introduced pooling of MCMC samples across burn-in iterations to improve empirical coverage of the ETA distribution. However, the BIC formula was passed the pooled count (N_pool = n_subjects × n_burnin_iters, typically 1600 for N=80, T=20) as the effective sample size N, instead of the true number of independent subjects. Because the log-likelihood term grows linearly with N_pool but ln(N) grows only logarithmically, the net effect is that the mixture model's likelihood advantage overwhelms the BIC complexity penalty, causing BIC to systematically over-select k (e.g., k=3 or k=4 for clearly unimodal data). | k selected too high (e.g., k=3/4 instead of k=1/2) for typical PK datasets, leading to spurious extra mixture components. | `fit_em_bic` now accepts an explicit `n_effective` argument (= n_subjects). The pooled log-likelihood is divided by T before computing BIC, and ln(n_subjects) is used for the BIC penalty. This is the standard IFM convention: the number of independent observations is the number of subjects, not the number of MCMC draws. |
| 13 | **Spurious "will be ignored" warnings for vine fit_options** — `omega_dist`, `mixture_components`, and `max_mixture_components` were not listed in the SAEM option allowlist (`method_specific_keys()` in `types.rs`). Any vine or vine-multimodal model therefore printed three "unknown fit_options key, will be ignored" warnings per run, even though the keys were handled correctly. | Noisy terminal output on every vine/vine-multimodal run; misleading to users who would think their vine settings were not applied. | Added all three keys to the SAEM allowlist. |

### Estimation improvements (June 2026)

A second audit pass corrected the M-step algorithm and added copula reporting.

| # | Change |
|---|--------|
| 14 | **Auto-BIC default for k**: `FitOptions::default()` now sets `saem_mixture_k = None` (auto-BIC), replacing the former hard-coded `Some(2)`. Unimodal data will select k=1 by default without user intervention. |
| 15 | **SA sufficient statistics for mixture marginals (NEED-2)**: The mixture marginal M-step now uses stochastic-approximation (SA) sufficient statistics (`S₀ⱼ, S₁ⱼ, S₂ⱼ` per component, damped by γ) instead of a full restart-EM fit at each SAEM iteration. This eliminates the non-stationarity and log-prior discontinuities that could disrupt the Markov chain when EM reconverged to a different ordering at each step. |
| 16 | **Pair-copula tree reporting for vine-multimodal (NEED-3)**: `vine_params` is now populated for `omega_dist = vine-multimodal`. The YAML `vine_mixture:` block and console output now include the full D-vine pair-copula tree structure (family, parameters, Kendall τ, tail-dependence coefficients) — previously these were only reported for `omega_dist = vine`. |
| 17 | **NaN-free YAML output (NEED-5 min viable)**: Mixture marginal SEs are not yet computed; the output layer now shows an explicit "SE: not available for mixture marginals" label rather than a bare `NaN` in the YAML/console output. |
| 18 | **Identifiability warnings (NICE-2)**: A warning is emitted in `result.warnings` when any mixture component has an effective subject count (N × wⱼ) below the threshold (default 10). Fires after BIC selection when auto-k is active. |
| 19 | **Simulation correctness (NICE-3)**: `sample()` now routes through `draw_eta()` (inverse Rosenblatt from the mixture-vine), fixing a silent regression where simulated VPC draws were taken from the Gaussian-equivalent OMEGA instead of the fitted mixture distribution. |

---

### What it is and why it matters

`omega_dist = vine` captures non-Gaussian *dependence* between ETAs while keeping each ETA marginally Gaussian. But sometimes the marginal distribution of a single ETA is itself non-Gaussian — most visibly, **bimodal**.

A bimodal ETA arises when the population naturally splits into two groups with different values of the same parameter: the textbook example is **CYP2D6 metaboliser status** in a population that mixes extensive and poor metabolisers. CYP2D6-metabolised drugs can show a CL ETA with a large mode (extensive) and a small mode (poor), separated by a two-to-five-fold difference. A single Gaussian centred between the two modes is a poor approximation: it assigns high prior probability to the intermediate region (which is actually sparse) and underestimates the tails of both modes.

In practice this misspecification causes:
- EBEs that "drift" toward the Gaussian centre, blurring the two subpopulations
- Biased θ estimates because the Laplace approximation integrates against the wrong prior
- CWRES that look heteroscedastic even after an adequate structural model
- VPC prediction intervals that are too narrow in the tails and too wide in the centre

`omega_dist = vine-multimodal` adds a **k-component Gaussian mixture** marginal (k ∈ {1,…,4}) for each ETA dimension on top of the D-vine copula. The mixture can represent any unimodal, skewed, or bimodal marginal. The dependence between ETAs is still described by pair-copulas — the two layers are estimated independently, so fitting a mixture marginal for CL does not perturb the CL–V dependence estimate.

---

### For pharmacometricians

**The intuition**

With `vine-multimodal`, each ETA now has these questions answered separately:

1. **Is there more than one mode, and how many?** — captured by the k−1 free mixture weights w₁,…,w_{k−1}, which tell you the fraction of subjects in each subpopulation (k chosen by BIC or fixed by `mixture_components`)
2. **How far apart are the subpopulations?** — captured by the component means μ₁,…,μₖ (on the η scale, sorted ascending)
3. **How spread out is each subpopulation?** — captured by the component SDs σ₁,…,σₖ
4. **How are the ETAs related to each other?** — still described by D-vine pair-copulas, exactly as in `omega_dist = vine`

The Gaussian-equivalent OMEGA diagonal (the standard between-subject variance) is still estimated and reported — it equals the overall marginal variance of the mixture — so all standard output continues to be interpretable.

**When should you use this?**

Use `vine-multimodal` when one or more of the following is true:

- EBE η histograms show two clearly separated bumps for one or more ETAs
- The drug is metabolised by a polymorphic enzyme (CYP2D6, CYP2C19, CYP2C9, NAT2, TPMT, …) and the dataset includes patients who were not genotyped
- A post-hoc analysis of your Gaussian fit shows a bimodal ETA distribution, but you cannot add genotype as a covariate because it was not collected
- The covariate you suspect drives the bimodality is unknown or unmeasured ("hidden covariate" problem)
- CWRES show two parallel bands across the PRED range — a tell-tale sign that the model is fitting a mix of two subpopulations as one

**When should you NOT use this?**

- When η histograms look unimodal — `vine-multimodal` will fit but adds unnecessary parameters; use `vine` instead
- When N < 50 (a warning is emitted): mixture components require enough subjects in each mode to be identifiable. A weak bimodality signal can appear as near-equal Gaussian components; the result is harmless but uninformative
- When the bimodality has a structural cause (wrong absorption model, flip-flop kinetics, misspecified compartment count) — fix the structural model first
- When the bimodality can be fully explained by a known covariate (WT, sex, genotype) — include the covariate instead; this gives a more interpretable model with one fewer distributional assumption

**Quick start**

```
[fit_options]
  method     = saem
  omega_dist = vine-multimodal
```

Let BIC choose how many subgroups each ETA needs (recommended when unsure):

```
[fit_options]
  method                 = saem
  omega_dist             = vine-multimodal
  mixture_components     = auto   # BIC selects k ∈ {1,2,3,4} per ETA after burn-in
  max_mixture_components = 4      # optional cap (default 4)
```

Force three components for every ETA:

```
[fit_options]
  method             = saem
  omega_dist         = vine-multimodal
  mixture_components = 3
```

**Note on method chaining**: `omega_dist = vine-multimodal` requires `method = saem`. Chaining with `focei` (e.g. `method = saem, focei`) is rejected at parse time because FOCEI's inner objective uses the Gaussian quadratic prior, which is incompatible with the mixture prior estimated during SAEM. For final inference use the SAEM result directly; the vine-corrected OFV is the appropriate quantity for model comparison.

**Choosing k**: by default k is selected by BIC (equivalent to `mixture_components = auto`). If you want to fix k for all ETAs, set `mixture_components = 2` (or 3, 4). See *Fit options reference* below.

**Important constraint**: `omega_dist = vine-multimodal` requires a **diagonal OMEGA**. The vine and mixture layers already capture all inter-ETA dependence through pair-copulas; declaring a free off-diagonal OMEGA element alongside them would double-count that dependence and make the model unidentified. If your current model file has `block_omega`, either remove the off-diagonal terms or use `gaussian` omega_dist.

**Reading the output**

The `vine_mixture:` section in the console and YAML reports, for each ETA:

- **n_components** — number of mixture components k selected for this ETA (k=1 means no multimodality detected — effectively Gaussian)
- **components[j].weight** — mixing weight wⱼ; values near 0 or 1 indicate a nearly unimodal distribution; components are always sorted ascending by mean
- **components[j].mu** — mean of component j on the η (log-transformed) scale
- **components[j].sd** — standard deviation of component j
- **mean, sd** — overall marginal mean and SD (these match the Gaussian-equivalent OMEGA diagonal)

On the η (log-transformed) scale, a difference of μ₂ − μ₁ ≈ ln(2) ≈ 0.69 corresponds to a two-fold difference in the untransformed parameter (e.g. a two-fold CL difference between poor and extensive metabolisers).

The sdtab adds per-subject columns `COMP_<ETA>` (most probable 1-based component index) and `PROB<j>_<ETA>` (posterior probability of component j) for each ETA with k ≥ 2.

Pair-copula output (family, parameters, Kendall τ, tail dependence) is identical to `omega_dist = vine`.

The `ΔOFV` value compares the mixture-vine prior to the Gaussian prior at the same EBEs. A large positive value indicates the mixture prior meaningfully improves fit; values close to zero suggest the population is not strongly bimodal on the ETA scale.

---

### For statisticians

**Model**

Each η_i margin follows a k-component Gaussian mixture (k ∈ {1,…,4}):

```
f_i(η) = Σⱼ wⱼ · φ(η; μⱼ_i, σⱼ_i)
F_i(η) = Σⱼ wⱼ · Φ((η − μⱼ_i) / σⱼ_i)
```

with identifiability constraint μ₁_i ≤ … ≤ μₖ_i enforced by label ordering. All mixing weights are clipped to [0.05, 1 − (k−1)·0.05] to prevent component collapse.

The PIT u_i = F_i(η_i) maps η to Uniform(0,1). These PITs are fed into the same D-vine pair-copula construction as in `omega_dist = vine`:

```
p(η₁, …, η_d) = ∏ᵢ f_i(η_i)  ×  c_vine(F₁(η₁), …, F_d(η_d))
```

The marginal component parameters (wⱼ, μⱼ_i, σⱼ_i for j = 1,…,k) and the copula parameters are separable and estimated in two sub-steps of the SAEM M-step.

**Estimation**

*E-step*: Metropolis-Hastings with componentwise proposals (one ETA at a time), using the mixture-vine joint prior log p(η₁,…,η_d). The component-wise proposal scale is set from the current mixture marginal SD.

*M-step*:
1. **Mixture marginals**: for each ETA dimension, update via stochastic-approximation (SA) sufficient statistics damped by γ. For each component j, the running statistics `S₀ⱼ ← (1−γ)S₀ⱼ + γ·r̄ⱼ`, `S₁ⱼ ← (1−γ)S₁ⱼ + γ·(r̄ⱼ·x̄ⱼ)`, `S₂ⱼ ← (1−γ)S₂ⱼ + γ·(r̄ⱼ·x̄²ⱼ)` accumulate responsibility-weighted sufficient statistics from the current SAEM samples. Component parameters are recovered as `wⱼ = S₀ⱼ/ΣS₀`, `μⱼ = S₁ⱼ/S₀ⱼ`, `σⱼ = √(S₂ⱼ/S₀ⱼ − μⱼ²)`. This is an approximate SA step, consistent with the SAEM Robbins–Monro framework, and avoids the log-prior discontinuities that full restart-EM can produce when it reconverges to a different component ordering.
2. **Gaussian-equivalent OMEGA**: update sample covariance via the standard SAEM stochastic-approximation formula (same as Gaussian SAEM). Used for the MH proposal scale and the OMEGA table in reports.
3. **Pair-copulas**: fit/re-fit the D-vine pair-copulas from the current PIT pseudo-observations using IFM (same as `omega_dist = vine`). Family selection occurs once (after burn-in) and is frozen.

**Vine-corrected OFV**

```
OFV_mixture = OFV_FOCE  +  2 × Σᵢ [ log p_mixture-vine(η̂ᵢ) − log p_Gauss(η̂ᵢ; Ω_equiv) ]
```

As with the `vine` path, the FOCE convention is applied: the `½d·log(2π)` constant that appears in the fully-normalized density is stripped before computing the OFV delta. This ensures the vine-corrected OFV is on the same scale as a standard FOCE OFV and the ΔOFV reflects only the genuine fit improvement from the non-Gaussian prior.

**AIC / BIC**

```
p  =  p_theta  +  p_Ω (free diagonal elements)  +  p_sigma
    + 3(k−1) × d      (extra params per ETA over a Gaussian marginal)
    + Σ_{pairs} n_params(family)
AIC = OFV_mixture + 2p
BIC = OFV_mixture + p × ln(N_obs)
```

Where k is the number of mixture components per ETA (k=1 adds no extra parameters; k=2 adds 3 per ETA — one weight, one extra mean, one extra SD; k=3 adds 6, etc.). The overall marginal mean and variance are already captured in k_theta and k_Ω respectively; `3(k−1)` counts only the *additional* shape parameters beyond a plain Gaussian marginal.

> **Note on the mixture mean**: the overall marginal mean E[η] = Σⱼ wⱼ μⱼ is expected to be near zero in a well-specified model (the theta captures the population-typical parameter). The mixture mean is not explicitly pinned to zero, which means it is technically a free parameter not counted in the penalty above. For a well-centred fit the effect is negligible; if the mixture mean drifts substantially from zero, the AIC/BIC penalty may be underestimated by 1 per ETA dimension.

**Simulation**

Draws from the fitted distribution use the inverse Rosenblatt transform identically to the `vine` path, except that the marginal inverse CDF step uses bisection on the mixture CDF rather than the Gaussian quantile function. Each draw requires O(d²) pair-copula h-inverse evaluations plus d bisection evaluations (≈30 iterations each); total cost is negligible compared to ODE evaluation.

**Marginal standard errors**

Standard errors for the mixture parameters (weights, means, SDs) per ETA are not yet implemented (Rung 4); these fields are absent from the YAML output. SEs for θ, Ω, σ are unaffected.

---

### Output reference (vine-multimodal additions)

Console output adds a `--- Vine Mixture ---` block after the OMEGA/SIGMA table. YAML output adds a `vine_mixture:` top-level key. Example (2 ETAs):

```yaml
vine_mixture:
  marginals:
    ETA_CL:
      n_components: 2
      components:
        - {weight: 0.312, mu: -0.891, sd: 0.183}   # component 1 = lower-mean (PM)
        - {weight: 0.688, mu:  0.104, sd: 0.201}   # component 2 = upper-mean (EM)
      mean: -0.194          # overall marginal mean (≈ 0 for well-specified model)
      sd:    0.524          # overall marginal SD (≈ sqrt of Gaussian-equiv OMEGA diagonal)
    ETA_V:
      n_components: 1
      components:
        - {weight: 1.000, mu: 0.001, sd: 0.196}    # k=1 selected by BIC → plain Gaussian
      mean:  0.001
      sd:    0.196
  pair_copulas:
    - tree: 1
      families:
        - Gaussian(rho=0.1821)
  ofv_foce_gaussian_prior:  1084.332
  ofv_vine_corrected:       1068.741    # use this for model comparison
  delta_ofv_mixture_advantage: 15.591  # positive = mixture-vine improves over Gaussian BSV
```

All standard output (theta, omega, sigma, AIC, BIC, sdtab, EBEs) is unchanged. AIC and BIC are computed from `ofv_vine_corrected` with the augmented parameter count (3(k−1) extra per ETA plus pair-copula params).

---

### Limitations

- **Marginal SEs**: standard errors for mixture parameters (weights, means, SDs) are not yet computed (Rung 4). These fields are absent from the YAML output.
- **Number of components**: k ∈ {1,…,4} per ETA, set with `mixture_components`. Automatic BIC selection is available with `mixture_components = auto`. The BIC selection operates on the SAEM chain samples pooled across burn-in iterations; with fewer than ~15 subjects per expected component the selection may be unreliable.
- **FOCEI chain**: `method = saem, focei` is rejected — `omega_dist = vine-multimodal` is SAEM-only. FOCEI's Gaussian quadratic prior is incompatible with the mixture prior.
- **EBEs under Gaussian prior**: final EBEs are optimised using the Gaussian-equivalent OMEGA (same as for `omega_dist = vine`), not the mixture prior. CWRES, IWRES, and NPDE are therefore computed at Gaussian-optimal EBEs. For strongly bimodal ETAs, subjects near the boundary between modes may show slightly biased EBEs.
- **Per-subject subgroup membership**: posterior membership probabilities per subject are reported in the sdtab (columns `COMP_<ETA>` = most probable 1-based component index, `PROB<j>_<ETA>` = posterior probability of component j). These probabilities are computed from the marginal mixture densities only; they do not account for the copula structure, so subjects near a mode boundary may show slightly inconsistent probabilities across correlated ETA dimensions.
- **Minimum N**: a warning is emitted when N < 50. Below this threshold the mixture components may not be separately identifiable from the SAEM chain samples.
- **Variable ordering and copula structure**: same limitations as `omega_dist = vine` (natural ordering, D-vine only, no R-vine).
- **Mixture initialisation**: the EM is initialised by splitting samples into k equal quantile groups. If the true modes are very close or the chain has not yet explored the full distribution (early iterations), the EM may converge to a near-Gaussian solution. This is self-correcting as the chain warms up.

---

### References

In addition to the vine copula references above:

- McLachlan, G.J., Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society B* **39**(1), 1–38.
- Credible pharmacometric background on CYP2D6 bimodality: Bertilsson, L. et al. (2002). Molecular genetics of CYP2D6. *British Journal of Clinical Pharmacology* **53**(2), 111–122.

---

## Data Format

Input data uses NONMEM-format CSV with columns:

- **Required**: `ID`, `TIME`, `DV`, `EVID`, `AMT`, `CMT`
- **Optional**: `RATE`, `MDV`, `II`, `SS`
- **Covariates**: Any additional columns are auto-detected

EVID codes: 0 = observation, 1 = dose, 4 = reset + dose.

## Examples

The `examples/` directory contains ready-to-run models:

| File | Description |
|------|-------------|
| `warfarin.ferx` | 1-compartment oral (warfarin PK) |
| `two_cpt_iv.ferx` | 2-compartment IV bolus |
| `two_cpt_oral_cov.ferx` | 2-compartment oral with covariates (WT, CRCL) |
| `mm_oral.ferx` | Michaelis-Menten elimination via ODE |
| `vine_tail_iv_gaussian.ferx` | Vine demo A — Gaussian block-Omega baseline (1-cpt IV) |
| `vine_tail_iv.ferx` | Vine demo A — `omega_dist = vine`; recovers Clayton lower-tail dependence |
| `vine_multimodal_iv_gaussian.ferx` | Vine demo B — Gaussian baseline for bimodal-CL scenario |
| `vine_multimodal_iv.ferx` | Vine demo B — `omega_dist = vine-multimodal`; detects bimodal ETA_CL |

Simulated datasets for the vine demos live in `data/`:

| File | Description |
|------|-------------|
| `data/vine_tail_iv.csv` | 80 subjects, dose=100, 10 time points; ETA_CL and ETA_V coupled via Clayton copula (τ≈0.40, λ_L≈0.59). Demonstrates asymmetric tail dependence that Gaussian Ω cannot capture. |
| `data/vine_multimodal_iv.csv` | 80 subjects, dose=100, 10 time points; ETA_CL bimodal (29% PM, 71% EM, CL fold-difference ≈2.7×). Demonstrates pharmacogenomic subpopulation structure a single Gaussian oversimplifies. |

The Python script that generated both datasets is in `examples/drafts/gen_vine_demos.py` and is fully seeded for reproducibility.

### Vine example datasets — step-by-step

**Step 1 — Fit all four models** (from the repository root):

```bash
# Scenario A: vine tail dependence
cargo run --release -- examples/vine_tail_iv_gaussian.ferx --data data/vine_tail_iv.csv
cargo run --release -- examples/vine_tail_iv.ferx          --data data/vine_tail_iv.csv

# Scenario B: vine-multimodal bimodal marginals
cargo run --release -- examples/vine_multimodal_iv_gaussian.ferx --data data/vine_multimodal_iv.csv
cargo run --release -- examples/vine_multimodal_iv.ferx           --data data/vine_multimodal_iv.csv
```

Each command writes two files to the current directory:
- `<model>-fit.yaml` — parameter estimates, SEs, AIC/BIC, vine copula/mixture details
- `<model>-sdtab.csv` — per-observation diagnostics (CWRES, IWRES, PRED, IPRED; plus `COMP_ETA_*` and `PROB*_ETA_*` columns for vine-multimodal)

Expected run times: 4–8 seconds each on a modern laptop.

**What to look for in `vine_tail_iv-fit.yaml`:**

```yaml
vine_copula:
  trees:
    - tree: 1
      pairs:
        - label: "ETA_CL ~ ETA_V"
          family: clayton          # non-Gaussian lower-tail dependence detected
          theta: 1.053             # Clayton parameter
          theta_se: 0.231
          kendall_tau: 0.345       # moderate positive concordance overall ...
          tail_dep_lower: 0.518    # ... but λ_L=0.52 in the lower tail
  ofv_vine_corrected: -2962.1      # corrected OFV (vs Gaussian -2957.2, ΔOFV=-4.9)
```

The `family: clayton` selection (rather than `gaussian`) and the non-zero `tail_dep_lower` are the key signals: the data contain asymmetric lower-tail concordance that a bivariate Gaussian cannot represent.

**What to look for in `vine_multimodal_iv-fit.yaml`:**

```yaml
vine_mixture:
  marginals:
    ETA_CL:
      n_components: 2              # BIC selected k=2 (bimodal)
      components:
        - {weight: 0.288, mu: -0.682, sd: 0.154}   # PM component
        - {weight: 0.712, mu: 0.274,  sd: 0.161}   # EM component
  ofv_vine_corrected: -3120.1      # vs Gaussian -3047.5 → ΔOFV = -72.6 (strong improvement)
  delta_ofv_mixture_advantage: 72.8
```

ΔOFV ≈ −73 against Gaussian (ΔAIC ≈ −65 after penalising 4 extra mixture parameters) is decisive evidence for the bimodal structure.

**Step 2 — Generate diagnostic plots with R**:

```bash
Rscript examples/vine_diagnostics.R
```

This reads the four output files from the current directory and produces:
- `vine_diagnostics_A.pdf` — 5-panel figure for Scenario A (tail dependence)
- `vine_diagnostics_B.pdf` — 5-panel figure for Scenario B (bimodal marginals)

Required R packages (`yaml`, `dplyr`, `ggplot2`, `patchwork`, `tidyr`, `scales`, `ggrepel`) are installed automatically if missing.

**What the R script shows and why it matters:**

*Scenario A panels:*

| Panel | What it shows | Key insight |
|-------|---------------|-------------|
| A-1 | EBE scatter (vine fit), coloured by lower/upper quartile membership | Lower-left cluster is denser than upper-right — the asymmetry is visible in the raw EBEs |
| A-2 | Kendall τ by tail region (Gaussian vs vine) | Vine shows τ_lower >> τ_upper; Gaussian shows τ_lower ≈ τ_upper — the Clayton asymmetry in numbers |
| A-3 | Simulated ETA pairs (Gaussian vs Clayton, n=2000) | Same marginals but different joint densities; lower-left concentration is unique to Clayton |
| A-4 | Bivariate chi-squared Q-Q plot | Under bivariate Gaussian, Mahalanobis D² ~ χ²(2); deviations below the diagonal signal more lower-tail mass than Gaussian predicts |
| A-5 | CWRES vs IPRED | Comparable residuals between fits — the OFV improvement comes from the prior, not data fit |

*Scenario B panels:*

| Panel | What it shows | Key insight |
|-------|---------------|-------------|
| B-1 | EBE_CL histogram with Gaussian vs bimodal density overlay | Two clear modes; Gaussian places high density in the sparse inter-modal region |
| B-2 | Posterior PM probability per subject (sorted) | Near-binary posteriors (P ≈ 0 or 1) confirm the two groups are well separated |
| B-3 | Individual IPRED profiles coloured by predicted metaboliser group | PMs (red) have flatter, higher profiles than EMs (green) — clinically meaningful subpopulations |
| B-4 | CWRES distribution: Gaussian vs vine-multimodal (violin plot) | Vine-multimodal narrows the CWRES distribution — the better prior produces better-calibrated individual predictions |
| B-5 | Simulated ETA_CL from both fitted models | Vine-multimodal simulation reproduces the two-peak structure; Gaussian produces a single wide hump |

**Interpreting the summary output:**

Running the script also prints model comparison tables and tail-concordance statistics to the console. The key numbers to cite in a report are:

- **Scenario A**: ΔOFV, Clayton θ, Kendall τ, λ_L (lower-tail dependence coefficient), and the empirical vs theoretical joint lower-tail probability.
- **Scenario B**: ΔOFV, k selected, mixture weights and component means, the fold-difference on the original CL scale (`exp(μ_EM) / exp(μ_PM)`), and the number of subjects classified in each group.

## R Package

An R wrapper package (`ferx`) provides `ferx_fit()`, `ferx_simulate()`, and `ferx_predict()` functions that call into this Rust engine via [extendr](https://extendr.github.io/). Source is at `../ferx`.

### Installation

```r
# Build the Rust backend and load the package
withr::with_dir("path/to/ferx", {
  system("cd src/rust && cargo build --release")
  devtools::load_all()
})
```

### Fitting a model

```r
result <- ferx_fit(
  model = "warfarin.ferx",
  data  = "warfarin.csv",
  method = "foce"        # or "focei"
)

result                   # prints summary with estimates and SEs
result$theta             # named vector of fixed-effect estimates
result$omega             # BSV covariance matrix
result$sigma             # residual error estimates
result$se_theta          # standard errors (NULL if covariance step failed)
result$sdtab             # data.frame with ID, TIME, DV, PRED, IPRED, CWRES, IWRES, ETA1..n
```

### Simulation and VPC

```r
sim <- ferx_simulate("warfarin.ferx", "warfarin.csv", n_sim = 100, seed = 42)
# Returns data.frame with SIM, ID, TIME, IPRED, DV_SIM

library(vpc)
obs <- read.csv("warfarin.csv")
vpc(obs = obs, sim = sim, sim_cols = list(dv = "DV_SIM"))
```

### Population predictions

```r
preds <- ferx_predict("warfarin.ferx", "warfarin.csv")
# Returns data.frame with ID, TIME, PRED (predictions at eta = 0)
```

## License

MIT — see [LICENSE](LICENSE).
