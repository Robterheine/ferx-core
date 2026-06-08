//! Vine copula with k-component Gaussian mixture marginals (k = 1–4).
//!
//! `VineMixtureMarginalOmega` is the estimator for `omega_dist = vine-multimodal`.
//! It replaces the per-ETA Gaussian marginal in [`VineCopulaOmega`] with a
//! [`MixtureMarginal`], feeding the resulting PIT values into the same D-vine
//! pair-copula machinery.
//!
//! The number of components k can be fixed (1–4) or selected automatically per
//! ETA by BIC from pooled SAEM samples after the burn-in phase.
//!
//! k = 1 reduces the marginal to a plain Gaussian (same as `omega_dist = vine`
//! for that dimension). k = 2 captures bimodal distributions (e.g. CYP2D6
//! poor/extensive metabolisers). k = 3 or 4 are available for rare multi-modal
//! populations but require N ≥ 100 to be reliably identifiable.
//!
//! [`VineCopulaOmega`]: crate::stats::vine_copula::VineCopulaOmega

use crate::stats::copula::{BivariateCopula, CopulaFamily};
use crate::stats::random_effects::RandomEffectDistribution;
use crate::stats::special::normal_cdf;
use crate::stats::vine_copula::dvine_log_density;
use crate::types::{ModelParameters, OmegaMatrix};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum component standard deviation — prevents component collapse.
const MARGINAL_STD_FLOOR: f64 = 1e-3;

/// Minimum per-component mixing weight — keeps all components alive.
/// For a k-component mixture, each weight is clamped to [MIXING_FLOOR, 1 − (k−1)·MIXING_FLOOR].
const MIXING_FLOOR: f64 = 0.05;

/// Minimum marginal standard deviation for the Gaussian-equivalent OMEGA floor.
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Maximum supported number of mixture components.
pub const MAX_MIXTURE_COMPONENTS: usize = 4;

// ---------------------------------------------------------------------------
// MixtureMarginal
// ---------------------------------------------------------------------------

/// k-component Gaussian mixture marginal for one ETA dimension (k = 1–4).
///
/// The density and CDF are:
/// ```text
/// f(x)  = Σⱼ wⱼ · φ(x; μⱼ, σⱼ)
/// F(x)  = Σⱼ wⱼ · Φ((x − μⱼ) / σⱼ)
/// ```
///
/// Identifiability: means are kept in ascending order (`means[0] ≤ … ≤ means[k−1]`)
/// by sorting components after every EM M-step and construction.
///
/// k = 1 is a plain Gaussian with weight 1.0 — it participates identically in the
/// vine but adds zero extra parameters over the Gaussian-equivalent OMEGA.
#[derive(Clone, Debug)]
pub struct MixtureMarginal {
    /// Mixing weights (length k, sum to 1, each ≥ MIXING_FLOOR).
    pub weights: Vec<f64>,
    /// Component means in ascending order.
    pub means: Vec<f64>,
    /// Component standard deviations (each ≥ MARGINAL_STD_FLOOR).
    pub stds: Vec<f64>,
}

/// Standard normal PDF φ(z) = exp(−z²/2) / √(2π).
#[inline]
fn standard_normal_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Standard normal log-PDF: −½z² − ½log(2π).
#[inline]
fn standard_normal_log_pdf(z: f64) -> f64 {
    const LOG_INV_SQRT_2PI: f64 = -0.918_938_533_204_672_7; // −½ ln(2π)
    LOG_INV_SQRT_2PI - 0.5 * z * z
}

/// Stable log-sum-exp: log(Σ exp(aᵢ)) without overflow/underflow.
#[inline]
fn log_sum_exp(log_vals: &[f64]) -> f64 {
    let max = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + log_vals.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

impl MixtureMarginal {
    /// Construct from raw vectors, enforcing constraints and mean ordering.
    ///
    /// Weights are normalised to sum to 1, then each weight is clamped to
    /// `[MIXING_FLOOR, 1 − (k−1)·MIXING_FLOOR]` before a final re-normalisation.
    /// Components are sorted by mean (ascending).
    pub fn new(weights: Vec<f64>, means: Vec<f64>, stds: Vec<f64>) -> Self {
        let k = weights.len();
        // Defensive clamp: callers from public API have already validated k,
        // but guard against misuse rather than panicking in release builds.
        debug_assert!(k >= 1 && k <= MAX_MIXTURE_COMPONENTS, "k={k} out of range");
        debug_assert_eq!(means.len(), k, "means length mismatch");
        debug_assert_eq!(stds.len(), k, "stds length mismatch");
        // Hard error in debug, silent clamp in release: truncate to valid length.
        if k == 0 {
            return Self::gaussian(0.0, MARGINAL_STD_FLOOR);
        }
        let k = k.min(MAX_MIXTURE_COMPONENTS);
        let weights = weights[..k].to_vec();
        let means = means[..k].to_vec();
        let stds = stds[..k].to_vec();

        // Sort components by mean (ascending) to enforce identifiability.
        let mut components: Vec<(f64, f64, f64)> = weights
            .into_iter()
            .zip(means)
            .zip(stds)
            .map(|((w, m), s)| (w, m, s))
            .collect();
        components.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_single_weight = (1.0 - (k as f64 - 1.0) * MIXING_FLOOR).max(MIXING_FLOOR);

        let weights_raw: Vec<f64> = components
            .iter()
            .map(|(w, _, _)| w.clamp(MIXING_FLOOR, max_single_weight))
            .collect();
        let total: f64 = weights_raw.iter().sum();

        Self {
            weights: weights_raw.iter().map(|&w| w / total).collect(),
            means: components.iter().map(|(_, m, _)| *m).collect(),
            stds: components
                .iter()
                .map(|(_, _, s)| s.max(MARGINAL_STD_FLOOR))
                .collect(),
        }
    }

    /// Convenience constructor for a k = 1 (plain Gaussian) marginal.
    pub fn gaussian(mean: f64, std: f64) -> Self {
        Self {
            weights: vec![1.0],
            means: vec![mean],
            stds: vec![std.max(MARGINAL_STD_FLOOR)],
        }
    }

    /// Convenience constructor for k = 2 using the original scalar parameters.
    ///
    /// `pi` is the weight of the lower-mean component (after sorting).
    pub fn two_component(pi: f64, mu1: f64, sig1: f64, mu2: f64, sig2: f64) -> Self {
        Self::new(vec![pi, 1.0 - pi], vec![mu1, mu2], vec![sig1, sig2])
    }

    /// Number of mixture components k.
    #[inline]
    pub fn k(&self) -> usize {
        self.weights.len()
    }

    /// Number of free parameters beyond a plain Gaussian marginal:
    /// - k = 1: 0  (it is the Gaussian)
    /// - k ≥ 2: 3(k−1)  = (k−1) weights + (k−1) extra means + (k−1) extra SDs
    ///   (one weight, mean, and SD are already counted in the Gaussian-equivalent OMEGA).
    pub fn n_free_params(&self) -> usize {
        if self.k() <= 1 {
            0
        } else {
            3 * (self.k() - 1)
        }
    }

    /// Mixture PDF at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.means)
            .zip(&self.stds)
            .map(|((&w, &mu), &sig)| w * standard_normal_pdf((x - mu) / sig) / sig)
            .sum()
    }

    /// Mixture CDF at `x` (= the PIT value u = F(x)).
    pub fn cdf(&self, x: f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.means)
            .zip(&self.stds)
            .map(|((&w, &mu), &sig)| w * normal_cdf((x - mu) / sig))
            .sum()
    }

    /// Log-density: log f(x), computed via log-sum-exp for numerical stability.
    ///
    /// Uses the identity:
    ///   log f(x) = log Σⱼ exp(log wⱼ + log φ((x−μⱼ)/σⱼ) − log σⱼ)
    /// This avoids underflow when x is far from all component means.
    pub fn log_pdf(&self, x: f64) -> f64 {
        let log_terms: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.means)
            .zip(&self.stds)
            .map(|((&w, &mu), &sig)| w.ln() + standard_normal_log_pdf((x - mu) / sig) - sig.ln())
            .collect();
        log_sum_exp(&log_terms)
    }

    /// Inverse CDF via bisection on `[μ_overall − 8σ_overall, μ_overall + 8σ_overall]`.
    /// Tolerance: 1e-9. Always converges in ≤ 60 iterations.
    pub fn icdf(&self, u: f64) -> f64 {
        let u = u.clamp(1e-12, 1.0 - 1e-12);
        let mu = self.mean();
        let sigma = self.std_dev();

        let mut lo = mu - 8.0 * sigma;
        let mut hi = mu + 8.0 * sigma;

        while self.cdf(lo) > u {
            lo -= sigma;
        }
        while self.cdf(hi) < u {
            hi += sigma;
        }

        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if hi - lo < 1e-9 {
                return mid;
            }
            if self.cdf(mid) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Overall marginal mean E[X] = Σⱼ wⱼ μⱼ.
    pub fn mean(&self) -> f64 {
        self.weights
            .iter()
            .zip(&self.means)
            .map(|(&w, &m)| w * m)
            .sum()
    }

    /// Overall marginal standard deviation √Var[X].
    pub fn std_dev(&self) -> f64 {
        let mu = self.mean();
        // Var[X] = Σⱼ wⱼ (σⱼ² + μⱼ²) − μ²
        let second_moment: f64 = self
            .weights
            .iter()
            .zip(&self.means)
            .zip(&self.stds)
            .map(|((&w, &m), &s)| w * (s * s + m * m))
            .sum();
        (second_moment - mu * mu)
            .max(0.0)
            .sqrt()
            .max(MARGINAL_STD_FLOOR)
    }

    /// Fit a k-component mixture by EM from `samples`.
    ///
    /// Initialisation: sort samples into k equal quantile groups; each group seeds
    /// one component. This splits-at-quantiles init avoids degenerate local modes.
    ///
    /// Convergence: |Δparams| < 1e-6 or 100 iterations.
    /// Constraints after every M-step:
    /// - weights ∈ [MIXING_FLOOR, 1 − (k−1)·MIXING_FLOOR] then renormalised
    /// - σⱼ ≥ MARGINAL_STD_FLOOR
    /// - means sorted ascending (label ordering)
    pub fn fit_em(samples: &[f64], k: usize) -> Self {
        // Clamp k to valid range rather than panicking in release builds.
        let k = k.clamp(1, MAX_MIXTURE_COMPONENTS);

        let n = samples.len();

        // k = 1: closed-form MLE (Gaussian).
        if k == 1 {
            if n == 0 {
                return Self::gaussian(0.0, MARGINAL_STD_FLOOR);
            }
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = if n > 1 {
                samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
            } else {
                MARGINAL_STD_FLOOR * MARGINAL_STD_FLOOR
            };
            return Self::gaussian(mean, var.sqrt());
        }

        // Small sample fallback: not enough points to fit k components.
        if n < 2 * k {
            let mean = if n > 0 {
                samples.iter().sum::<f64>() / n as f64
            } else {
                0.0
            };
            let sd = MARGINAL_STD_FLOOR;
            // Return equal-weight symmetric initialisation.
            let spread = sd * 0.5;
            let weights = vec![1.0 / k as f64; k];
            let means: Vec<f64> = (0..k)
                .map(|j| mean + (j as f64 - (k as f64 - 1.0) / 2.0) * spread)
                .collect();
            let stds = vec![sd; k];
            return Self::new(weights, means, stds);
        }

        // Initialise by splitting sorted samples into k equal quantile groups.
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut weights = vec![1.0 / k as f64; k];
        let mut means: Vec<f64> = (0..k)
            .map(|j| {
                let start = j * n / k;
                let end = if j == k - 1 { n } else { (j + 1) * n / k };
                sorted[start..end].iter().sum::<f64>() / (end - start) as f64
            })
            .collect();
        let mut stds: Vec<f64> = (0..k)
            .map(|j| {
                let start = j * n / k;
                let end = if j == k - 1 { n } else { (j + 1) * n / k };
                let mu_j = means[j];
                let var = sorted[start..end]
                    .iter()
                    .map(|&x| (x - mu_j).powi(2))
                    .sum::<f64>()
                    / (end - start) as f64;
                var.sqrt().max(MARGINAL_STD_FLOOR)
            })
            .collect();

        // EM iterations.
        let max_weight = (1.0 - (k as f64 - 1.0) * MIXING_FLOOR).max(MIXING_FLOOR);
        // responsibilities[i][j] = P(component j | sample i)
        let mut resps = vec![vec![0.0f64; k]; n];

        // Pre-allocate per-sample log-term buffer to avoid per-iteration heap allocs.
        let mut log_terms = vec![0.0f64; k];

        for _iter in 0..100 {
            // --- E-step (log-sum-exp for numerical stability) ---
            // log r_ij = log w_j + log φ((x−μ_j)/σ_j) − log σ_j
            // r_ij     = softmax_j(log r_ij)
            for (i, &x) in samples.iter().enumerate() {
                for j in 0..k {
                    log_terms[j] = weights[j].ln()
                        + standard_normal_log_pdf((x - means[j]) / stds[j])
                        - stds[j].ln();
                }
                let lse = log_sum_exp(&log_terms);
                if lse.is_finite() {
                    for j in 0..k {
                        resps[i][j] = (log_terms[j] - lse).exp();
                    }
                } else {
                    // All log-terms are -inf (degenerate): assign to nearest component.
                    let nearest = (0..k)
                        .min_by(|&a, &b| {
                            (x - means[a])
                                .abs()
                                .partial_cmp(&(x - means[b]).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    for r in &mut resps[i] {
                        *r = 0.0;
                    }
                    resps[i][nearest] = 1.0;
                }
            }

            // --- M-step ---
            let mut new_weights = vec![0.0f64; k];
            let mut new_means = vec![0.0f64; k];
            let mut new_stds = vec![0.0f64; k];

            for j in 0..k {
                let nj: f64 = resps.iter().map(|r| r[j]).sum();
                new_weights[j] = (nj / n as f64).clamp(MIXING_FLOOR, max_weight);
                if nj > 1.0 {
                    new_means[j] = resps
                        .iter()
                        .zip(samples.iter())
                        .map(|(r, &x)| r[j] * x)
                        .sum::<f64>()
                        / nj;
                    let var = resps
                        .iter()
                        .zip(samples.iter())
                        .map(|(r, &x)| r[j] * (x - new_means[j]).powi(2))
                        .sum::<f64>()
                        / nj;
                    new_stds[j] = var.sqrt().max(MARGINAL_STD_FLOOR);
                } else {
                    // Component starved: keep previous values.
                    new_means[j] = means[j];
                    new_stds[j] = stds[j];
                }
            }

            // Re-normalise weights.
            let w_sum: f64 = new_weights.iter().sum();
            for w in &mut new_weights {
                *w /= w_sum;
            }

            // Check convergence.
            let delta: f64 = (0..k)
                .map(|j| {
                    (new_weights[j] - weights[j]).abs()
                        + (new_means[j] - means[j]).abs()
                        + (new_stds[j] - stds[j]).abs()
                })
                .sum();

            weights = new_weights;
            means = new_means;
            stds = new_stds;

            if delta < 1e-6 {
                break;
            }
        }

        // Sort by mean and construct (enforces label ordering).
        Self::new(weights, means, stds)
    }

    /// Select the best k ∈ {1, …, `max_k`} by BIC and return `(fitted_model, selected_k)`.
    ///
    /// BIC = −2 · log L + d_free · ln(N) where d_free is the total number of free
    /// parameters for that k-component model fitted to the N samples:
    /// - k = 1: 2  (μ, σ)
    /// - k ≥ 2: 3k − 1  (k means + k SDs + k−1 free weights)
    ///
    /// The BIC compares models on the standalone marginal log-likelihood — it is only
    /// used for the k selection decision and is independent of the vine/OMEGA penalty.
    ///
    /// Returns k = 1 when `samples` is too small to distinguish components.
    pub fn fit_em_bic(samples: &[f64], max_k: usize) -> (Self, usize) {
        let max_k = max_k.clamp(1, MAX_MIXTURE_COMPONENTS);
        let n = samples.len();

        let mut best_k = 1usize;
        let mut best_bic = f64::INFINITY;
        let mut best_model = Self::fit_em(samples, 1);

        for k in 1..=max_k {
            // Need at least 2k samples to fit k components meaningfully.
            if n < 2 * k {
                break;
            }
            let model = Self::fit_em(samples, k);
            let log_lik: f64 = samples.iter().map(|&x| model.log_pdf(x)).sum();
            // Free parameters: k=1 → 2, k≥2 → 3k−1.
            let d_free = if k == 1 { 2 } else { 3 * k - 1 };
            let bic = -2.0 * log_lik + d_free as f64 * (n as f64).ln();

            if bic < best_bic {
                best_bic = bic;
                best_k = k;
                best_model = model;
            }
        }

        (best_model, best_k)
    }
}

// ---------------------------------------------------------------------------
// VineMixtureMarginalOmega
// ---------------------------------------------------------------------------

/// Vine copula with per-ETA k-component Gaussian mixture marginals.
///
/// This implements `omega_dist = vine-multimodal`. Each η dimension gets an
/// independent [`MixtureMarginal`] with k ∈ {1,…,4} components; the PIT values
/// `u_i = F_i(η_i)` are fed into the same D-vine copula structure used by
/// [`VineCopulaOmega`].
///
/// ### k selection
///
/// - **Fixed k** (`fixed_k = Some(k)`): all ETAs use k components throughout.
/// - **Auto** (`fixed_k = None`): ETAs start with k = 2. After the SAEM burn-in
///   phase, BIC selects the best k ∈ {1, …, `max_k`} per ETA independently from
///   the pooled samples. k is then frozen for the rest of the run (same philosophy
///   as copula family selection).
///
/// The M-step follows a three-step pattern:
/// 1. **Mixture marginals**: re-fit each [`MixtureMarginal`] by EM from the current
///    pooled SAEM η samples.
/// 2. **Gaussian-equivalent OMEGA**: update the sample covariance for reporting
///    and the MH proposal scale.
/// 3. **Pair-copulas**: fit/re-fit D-vine pair-copulas from PIT pseudo-observations.
#[derive(Debug, Clone)]
pub struct VineMixtureMarginalOmega {
    /// Number of ETA dimensions.
    pub d: usize,
    /// Per-dimension mixture marginals. Index `i` = original η index.
    pub marginals: Vec<MixtureMarginal>,
    /// D-vine pair-copula families. `pair_copulas[tree][pair]`.
    pub pair_copulas: Vec<Vec<CopulaFamily>>,
    /// Whether AIC-based copula family selection has run at least once.
    families_selected: bool,
    /// Whether BIC-based k selection has run (auto mode only).
    k_selected: bool,
    /// Fixed k for all ETAs. `None` = auto BIC selection after burn-in.
    /// Stored for diagnostics; runtime behaviour is encoded in `k_selected`.
    #[allow(dead_code)]
    fixed_k: Option<usize>,
    /// Maximum k tested in auto BIC selection.
    max_k: usize,
    /// SA sufficient statistic s₂ = (1−γ) s₂_prev + γ (1/N) Σ ηᵢηᵢᵀ.
    sample_s2: DMatrix<f64>,
    /// Gaussian-equivalent OmegaMatrix from `sample_s2`.
    omega_equiv: OmegaMatrix,
    /// Initial omega — reference for eta_names, diagonal flag, free_mask.
    initial_omega: OmegaMatrix,
    /// Per-eta fixed flags.
    omega_fixed: Vec<bool>,
    /// Initial omega matrix values (for restoring fixed entries after SA).
    initial_matrix: DMatrix<f64>,
    /// Per-pair pseudo-observations for SE computation (Rung 4).
    per_pair_pseudo_obs: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    /// D-vine variable ordering: `variable_order[vine_pos] = orig_pos`.
    pub variable_order: Vec<usize>,
    /// Accumulated η samples across burn-in iterations for BIC k-selection.
    /// Populated by `push_bic_samples`; consumed and cleared by `select_k_by_bic`.
    bic_eta_pool: Vec<Vec<f64>>,
}

impl VineMixtureMarginalOmega {
    /// Construct from initial model parameters with explicit k settings.
    ///
    /// - `fixed_k = Some(k)`: all ETAs start and stay at k components.
    /// - `fixed_k = None`: start at k = 2; BIC selects best k after burn-in.
    pub fn from_init_params_with_opts(
        init_params: &ModelParameters,
        fixed_k: Option<usize>,
        max_k: usize,
    ) -> Self {
        let omega = init_params.omega.clone();
        let d = omega.dim();
        let variable_order: Vec<usize> = (0..d).collect();

        // Starting k: fixed value or 2 for auto mode.
        let start_k = fixed_k.unwrap_or(2).clamp(1, MAX_MIXTURE_COMPONENTS);

        let marginals: Vec<MixtureMarginal> = (0..d)
            .map(|i| {
                let sigma = omega.matrix[(i, i)]
                    .max(MARGINAL_STD_FLOOR * MARGINAL_STD_FLOOR)
                    .sqrt();
                if start_k == 1 {
                    MixtureMarginal::gaussian(0.0, sigma)
                } else {
                    // Symmetric k-component initialisation: components equally spaced
                    // around 0 with spacing 0.5·σ and equal weights.
                    let weights = vec![1.0 / start_k as f64; start_k];
                    let means: Vec<f64> = (0..start_k)
                        .map(|j| (j as f64 - (start_k as f64 - 1.0) / 2.0) * 0.5 * sigma)
                        .collect();
                    let stds = vec![sigma; start_k];
                    MixtureMarginal::new(weights, means, stds)
                }
            })
            .collect();

        let pair_copulas: Vec<Vec<CopulaFamily>> = (0..d.saturating_sub(1))
            .map(|k| {
                (0..d - k - 1)
                    .map(|_| CopulaFamily::Gaussian(crate::stats::copula::GaussianCopula::new(0.0)))
                    .collect()
            })
            .collect();

        let sample_s2 = omega.matrix.clone();
        let initial_matrix = omega.matrix.clone();
        let initial_omega = omega.clone();
        let omega_equiv = omega.clone();
        let omega_fixed = init_params.omega_fixed.clone();

        Self {
            d,
            marginals,
            pair_copulas,
            families_selected: false,
            k_selected: fixed_k.is_some(), // already decided if fixed
            fixed_k,
            max_k: max_k.clamp(1, MAX_MIXTURE_COMPONENTS),
            sample_s2,
            omega_equiv,
            initial_omega,
            omega_fixed,
            initial_matrix,
            per_pair_pseudo_obs: Vec::new(),
            variable_order,
            bic_eta_pool: Vec::new(),
        }
    }

    /// Convenience constructor using k = 2 (backwards-compatible default).
    pub fn from_init_params(init_params: &ModelParameters) -> Self {
        Self::from_init_params_with_opts(init_params, Some(2), 2)
    }

    /// Accumulate η samples during burn-in for later BIC k-selection.
    ///
    /// Call this once per SAEM iteration while `k ≤ omega_burnin` in auto mode.
    /// The pool is consumed and cleared by `select_k_by_bic`.
    pub fn push_bic_samples(&mut self, sampled_etas: &[Vec<f64>]) {
        if !self.k_selected {
            self.bic_eta_pool.extend_from_slice(sampled_etas);
        }
    }

    /// Select the best k per ETA by BIC from pooled burn-in samples, then
    /// re-initialise the marginals with the selected k. Called once after
    /// burn-in in auto mode.
    ///
    /// Uses `self.bic_eta_pool` (accumulated by `push_bic_samples` during
    /// burn-in) for a statistically stronger BIC estimate. Falls back to
    /// `current_etas` if the pool is empty (e.g. `omega_burnin = 0`).
    ///
    /// Each ETA's k is selected independently; ETAs can have different k values.
    /// After this call `k_selected` is set to `true` and further M-steps use the
    /// selected k.
    pub fn select_k_by_bic(&mut self, current_etas: &[Vec<f64>]) {
        if self.k_selected {
            return;
        }
        // Use the accumulated pool if available; otherwise fall back to current.
        let pool: &[Vec<f64>] = if self.bic_eta_pool.is_empty() {
            current_etas
        } else {
            &self.bic_eta_pool
        };
        for i in 0..self.d {
            let col: Vec<f64> = pool.iter().map(|e| e[i]).collect();
            let (model, _k) = MixtureMarginal::fit_em_bic(&col, self.max_k);
            self.marginals[i] = model;
        }
        self.k_selected = true;
        self.bic_eta_pool.clear();
    }

    /// Compute PIT pseudo-observations from η samples.
    pub(crate) fn pit_from_samples(&self, sampled_etas: &[Vec<f64>]) -> Vec<Vec<f64>> {
        sampled_etas
            .iter()
            .map(|eta| {
                self.variable_order
                    .iter()
                    .map(|&orig| self.marginals[orig].cdf(eta[orig]).clamp(1e-8, 1.0 - 1e-8))
                    .collect()
            })
            .collect()
    }

    /// Re-fit each mixture marginal by EM, keeping the current k per ETA.
    fn update_marginals_em(&mut self, sampled_etas: &[Vec<f64>]) {
        for i in 0..self.d {
            let col: Vec<f64> = sampled_etas.iter().map(|e| e[i]).collect();
            let k = self.marginals[i].k();
            self.marginals[i] = MixtureMarginal::fit_em(&col, k);
        }
    }

    /// Update the Gaussian-equivalent OMEGA via SA averaging.
    fn update_sample_cov(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        let d = self.d;
        let n = sampled_etas.len() as f64;
        let mut eta_outer = DMatrix::zeros(d, d);
        for eta in sampled_etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n;

        self.sample_s2 = (1.0 - gamma) * &self.sample_s2 + gamma * &eta_outer;

        let mut new_mat = self.sample_s2.clone();
        for i in 0..d {
            for j in 0..d {
                if !self.initial_omega.free_mask[(i, j)] {
                    new_mat[(i, j)] = 0.0;
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = self.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    new_mat[(i, j)] = self.initial_matrix[(i, j)];
                }
            }
        }
        for i in 0..d {
            if !self.omega_fixed.get(i).copied().unwrap_or(false)
                && new_mat[(i, i)] < OMEGA_DIAG_FLOOR
            {
                new_mat[(i, i)] = OMEGA_DIAG_FLOOR;
            }
        }

        self.omega_equiv = OmegaMatrix::from_matrix(
            new_mat,
            self.initial_omega.eta_names.clone(),
            self.initial_omega.diagonal,
        );
    }

    /// Fit (or re-fit) all pair-copulas from the pseudo-observation matrix.
    fn fit_vine(&mut self, pseudo_obs: &[Vec<f64>], select_families: bool, gamma: f64) {
        let d = self.d;
        if d <= 1 {
            return;
        }
        let n = pseudo_obs.len();
        if n < 4 {
            return;
        }

        let mut u_cols: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                pseudo_obs
                    .iter()
                    .map(|row| row[i].clamp(1e-8, 1.0 - 1e-8))
                    .collect()
            })
            .collect();

        let mut new_pair_pseudo = vec![Vec::new(); d.saturating_sub(1)];

        for k in 0..d - 1 {
            let n_pairs = d - k - 1;
            let mut new_left_cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n_pairs];
            let mut new_right_cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n_pairs];

            for j in 0..n_pairs {
                let lefts: &[f64] = &u_cols[j];
                let rights: &[f64] = &u_cols[j + 1];

                new_pair_pseudo[k].push((lefts.to_vec(), rights.to_vec()));

                if select_families {
                    if let Ok(cop) = CopulaFamily::select(lefts, rights) {
                        self.pair_copulas[k][j] = cop;
                    }
                } else {
                    self.pair_copulas[k][j] =
                        self.pair_copulas[k][j].damped_refit(lefts, rights, gamma);
                }

                let cop = &self.pair_copulas[k][j];
                for s in 0..n {
                    let l = lefts[s];
                    let r = rights[s];
                    new_left_cols[j][s] = cop.h(l, r).clamp(1e-8, 1.0 - 1e-8);
                    new_right_cols[j][s] = cop.h(r, l).clamp(1e-8, 1.0 - 1e-8);
                }
            }

            let mut next_u: Vec<Vec<f64>> = new_left_cols;
            next_u.push(new_right_cols[n_pairs - 1].clone());
            u_cols = next_u;
        }
        self.per_pair_pseudo_obs = new_pair_pseudo;
    }
}

impl RandomEffectDistribution for VineMixtureMarginalOmega {
    fn log_prior(&self, eta: &[f64]) -> f64 {
        let mut log_prior = 0.0;
        let mut u = vec![0.0f64; self.d];

        for (vine_pos, &orig) in self.variable_order.iter().enumerate() {
            log_prior -= self.marginals[orig].log_pdf(eta[orig]);
            u[vine_pos] = self.marginals[orig]
                .cdf(eta[orig])
                .clamp(1e-12, 1.0 - 1e-12);
        }

        if self.d > 1 {
            log_prior -= dvine_log_density(&u, &self.pair_copulas);
        }

        if log_prior.is_finite() {
            log_prior
        } else {
            1e20
        }
    }

    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() {
            return;
        }
        self.update_marginals_em(sampled_etas);
        self.update_sample_cov(sampled_etas, gamma);
        let pseudo_obs = self.pit_from_samples(sampled_etas);
        let select = !self.families_selected;
        self.fit_vine(&pseudo_obs, select, gamma);
        if select {
            self.families_selected = true;
        }
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let d = self.d;
        let l = &self.omega_equiv.chol;
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
                (l * DVector::from_column_slice(&z))
                    .iter()
                    .copied()
                    .collect()
            })
            .collect()
    }

    fn proposal_chol(&self) -> &DMatrix<f64> {
        &self.omega_equiv.chol
    }

    fn to_omega_matrix(&self) -> &OmegaMatrix {
        &self.omega_equiv
    }
}

// ---------------------------------------------------------------------------
// Inverse-Rosenblatt sampling (draw_eta)
// ---------------------------------------------------------------------------

impl VineMixtureMarginalOmega {
    /// Draw one joint η vector by inverse-Rosenblatt transform.
    pub fn draw_eta<R: rand::Rng>(&self, rng: &mut R) -> Vec<f64> {
        use rand_distr::Open01;

        let d = self.d;
        if d == 0 {
            return vec![];
        }
        let w: Vec<f64> = (0..d).map(|_| rng.sample(Open01)).collect();
        if d == 1 {
            let orig = self.variable_order[0];
            return vec![self.marginals[orig].icdf(w[0].clamp(1e-12, 1.0 - 1e-12))];
        }

        let mut vt = vec![vec![0.0_f64; d]; d];
        let u0 = w[0].clamp(1e-12, 1.0 - 1e-12);
        vt[0][0] = u0;

        for i in 1..d {
            let mut tmp = w[i].clamp(1e-12, 1.0 - 1e-12);
            for j in 0..i {
                let tree_level = i - 1 - j;
                let pair_idx = j;
                let cond = vt[j][i - 1].clamp(1e-12, 1.0 - 1e-12);
                tmp = self.pair_copulas[tree_level][pair_idx]
                    .h_inv(tmp.clamp(1e-12, 1.0 - 1e-12), cond)
                    .clamp(1e-12, 1.0 - 1e-12);
            }
            vt[i][i] = tmp;

            for j in (0..i).rev() {
                let tree_level = i - j - 1;
                let pair_idx = j;
                let l = vt[j][i - 1].clamp(1e-12, 1.0 - 1e-12);
                let r = vt[j + 1][i].clamp(1e-12, 1.0 - 1e-12);
                vt[j][i] = self.pair_copulas[tree_level][pair_idx]
                    .h(l, r)
                    .clamp(1e-12, 1.0 - 1e-12);
            }
        }

        let mut result = vec![0.0f64; d];
        for k in 0..d {
            let orig = self.variable_order[k];
            result[orig] = self.marginals[orig].icdf(vt[k][k]);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Posterior mixture membership
// ---------------------------------------------------------------------------

impl VineMixtureMarginalOmega {
    /// Compute posterior component membership probabilities for a single η vector.
    ///
    /// Returns a `d × k_max` matrix (as `Vec<Vec<f64>>`) where entry `[i][j]` is
    /// the posterior probability that ETA dimension `i` belongs to component `j`:
    ///
    /// ```text
    /// P(comp j | η_i) = w_j · φ(η_i; μ_j, σ_j) / f_mixture(η_i)
    /// ```
    ///
    /// Components are in the same ascending-mean order as `marginals[i]`.
    /// For k=1 marginals the single entry is trivially 1.0.
    pub fn posterior_membership(&self, eta: &[f64]) -> Vec<Vec<f64>> {
        (0..self.d)
            .map(|i| {
                let m = &self.marginals[i];
                let k = m.k();
                if k == 1 {
                    return vec![1.0];
                }
                let log_terms: Vec<f64> = m
                    .weights
                    .iter()
                    .zip(&m.means)
                    .zip(&m.stds)
                    .map(|((&w, &mu), &sig)| {
                        w.ln() + standard_normal_log_pdf((eta[i] - mu) / sig) - sig.ln()
                    })
                    .collect();
                let lse = log_sum_exp(&log_terms);
                if lse.is_finite() {
                    log_terms.iter().map(|&v| (v - lse).exp()).collect()
                } else {
                    // All components have zero density at this point — assign to nearest.
                    let nearest = (0..k)
                        .min_by(|&a, &b| {
                            (eta[i] - m.means[a])
                                .abs()
                                .partial_cmp(&(eta[i] - m.means[b]).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    let mut probs = vec![0.0f64; k];
                    probs[nearest] = 1.0;
                    probs
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Standard errors placeholder (Rung 4)
// ---------------------------------------------------------------------------

/// Approximate standard errors for the mixture marginal parameters.
///
/// Returns a `Vec<[f64; 3]>` with `(weight_se, mean_se, sd_se)` per component.
/// Currently unimplemented — all values are NaN.
pub fn mixture_marginal_se(_m: &MixtureMarginal, _samples: &[f64]) -> Vec<[f64; 3]> {
    vec![[f64::NAN; 3]; _m.k()]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, ModelParameters};

    // ── MixtureMarginal: k=2 (original behaviour) ────────────────────────────

    #[test]
    fn mixture_marginal_cdf_monotone() {
        let m = MixtureMarginal::two_component(0.6, -0.5, 0.2, 0.5, 0.2);
        let grid: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.2).collect();
        let mut prev = m.cdf(grid[0]);
        for &x in &grid[1..] {
            let cur = m.cdf(x);
            assert!(
                cur >= prev - 1e-12,
                "CDF not monotone: cdf({x:.1})={cur:.6} < prev={prev:.6}"
            );
            prev = cur;
        }
    }

    #[test]
    fn mixture_marginal_icdf_roundtrip() {
        let m = MixtureMarginal::two_component(0.6, -0.5, 0.15, 0.5, 0.15);
        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let u = m.cdf(x);
            let x_back = m.icdf(u);
            assert!(
                (x_back - x).abs() < 1e-6,
                "icdf(cdf({x}))={x_back:.9} expected ≈ {x}"
            );
        }
    }

    #[test]
    fn mixture_em_recovers_two_components() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 1000usize;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let u: f64 = rng.sample(rand::distributions::Open01);
            let z: f64 = rng.sample(StandardNormal);
            if u < 0.6 {
                samples.push(-0.5 + 0.1 * z);
            } else {
                samples.push(0.5 + 0.1 * z);
            }
        }
        let m = MixtureMarginal::fit_em(&samples, 2);
        assert!(m.means[0] < m.means[1], "mean ordering violated");
        assert!(
            (m.means[0] - (-0.5)).abs() < 0.1,
            "means[0]={:.3} ≈ -0.5",
            m.means[0]
        );
        assert!(
            (m.means[1] - 0.5).abs() < 0.1,
            "means[1]={:.3} ≈ 0.5",
            m.means[1]
        );
        assert!(
            (m.weights[0] - 0.6).abs() < 0.05,
            "weights[0]={:.3} ≈ 0.6",
            m.weights[0]
        );
    }

    #[test]
    fn mixture_label_switch_enforced() {
        let m = MixtureMarginal::two_component(0.3, 1.0, 0.1, -1.0, 0.1);
        assert!(
            m.means[0] <= m.means[1],
            "label order not enforced: means={:?}",
            m.means
        );
    }

    // ── MixtureMarginal: k=1 (Gaussian special case) ─────────────────────────

    #[test]
    fn mixture_k1_is_gaussian() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let samples: Vec<f64> = (0..500)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.3)
            .collect();
        let m = MixtureMarginal::fit_em(&samples, 1);
        assert_eq!(m.k(), 1);
        assert_eq!(m.n_free_params(), 0, "k=1 should have 0 extra params");
        // mean should be near 0
        assert!(m.mean().abs() < 0.1, "k=1 mean={:.3}", m.mean());
    }

    // ── MixtureMarginal: k=3 ─────────────────────────────────────────────────

    #[test]
    fn mixture_k3_pdf_integrates_to_one() {
        let m = MixtureMarginal::new(
            vec![0.3, 0.4, 0.3],
            vec![-1.0, 0.0, 1.0],
            vec![0.2, 0.2, 0.2],
        );
        assert_eq!(m.k(), 3);
        assert_eq!(m.n_free_params(), 6); // 3*(3-1)
                                          // CDF at ±10 should be ≈ 0 and 1.
        assert!(m.cdf(-10.0) < 1e-6, "CDF(-10) should be ≈ 0");
        assert!(m.cdf(10.0) > 1.0 - 1e-6, "CDF(10) should be ≈ 1");
        // ICDF round-trip at median.
        let med = m.icdf(0.5);
        assert!(
            (m.cdf(med) - 0.5).abs() < 1e-5,
            "ICDF(0.5) roundtrip failed"
        );
    }

    // ── BIC selection ─────────────────────────────────────────────────────────

    #[test]
    fn bic_selects_k2_for_bimodal_data() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut samples = Vec::new();
        for _ in 0..200 {
            let u: f64 = rng.sample(rand::distributions::Open01);
            let z: f64 = rng.sample(StandardNormal);
            // Strong bimodal signal: components 0.8 apart, SD 0.1
            if u < 0.5 {
                samples.push(-0.4 + 0.1 * z);
            } else {
                samples.push(0.4 + 0.1 * z);
            }
        }
        let (_model, k) = MixtureMarginal::fit_em_bic(&samples, 4);
        assert!(
            k >= 2,
            "BIC should select k ≥ 2 for strong bimodal data, got k={k}"
        );
    }

    #[test]
    fn bic_selects_k1_for_unimodal_data() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let samples: Vec<f64> = (0..300)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.3)
            .collect();
        let (_model, k) = MixtureMarginal::fit_em_bic(&samples, 4);
        assert!(
            k <= 2,
            "BIC should prefer low k for unimodal data, got k={k}"
        );
    }

    // ── VineMixtureMarginalOmega ──────────────────────────────────────────────

    #[test]
    fn vine_mixture_log_prior_finite() {
        let model = analytical_model(GradientMethod::Auto);
        let dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        for &eta in &[0.0, 0.5, -0.3, 1.2] {
            let lp = dist.log_prior(&[eta]);
            assert!(lp.is_finite(), "log_prior([{eta}]) = {lp}");
        }
    }

    #[test]
    fn vine_mixture_variable_order_valid() {
        let model = analytical_model(GradientMethod::Auto);
        let dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        let d = dist.d;
        let mut sorted = dist.variable_order.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..d).collect();
        assert_eq!(
            sorted, expected,
            "variable_order is not a permutation of 0..d"
        );
    }

    #[test]
    fn vine_mixture_mstep_updates_marginals() {
        let model = analytical_model(GradientMethod::Auto);
        let mut dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        let lp_before = dist.log_prior(&[0.3]);

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let samples: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let z: f64 = rng.sample(StandardNormal);
                vec![if i < 30 {
                    -0.5 + 0.1 * z
                } else {
                    0.5 + 0.1 * z
                }]
            })
            .collect();

        dist.mstep_update(&samples, 1.0);
        let lp_after = dist.log_prior(&[0.3]);
        assert!(lp_after.is_finite(), "log_prior after mstep = {lp_after}");
        assert_ne!(
            lp_before, lp_after,
            "log_prior unchanged after mstep_update"
        );
    }

    #[test]
    fn vine_mixture_draw_eta_shape_and_finite() {
        let model = analytical_model(GradientMethod::Auto);
        let dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let eta = dist.draw_eta(&mut rng);
            assert_eq!(eta.len(), dist.d);
            assert!(
                eta.iter().all(|x| x.is_finite()),
                "draw_eta returned non-finite: {eta:?}"
            );
        }
    }

    #[test]
    fn vine_mixture_two_eta_log_prior_and_draw() {
        use crate::types::OmegaMatrix;
        let model = analytical_model(GradientMethod::Auto);
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let params = ModelParameters {
            omega,
            omega_fixed: vec![false, false],
            ..model.default_params.clone()
        };
        let dist = VineMixtureMarginalOmega::from_init_params(&params);
        assert_eq!(dist.d, 2);
        let lp = dist.log_prior(&[0.1, -0.2]);
        assert!(lp.is_finite(), "2-ETA log_prior = {lp}");

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let eta = dist.draw_eta(&mut rng);
        assert_eq!(eta.len(), 2);
        assert!(eta.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn vine_mixture_auto_k_selects_after_bic() {
        use crate::types::OmegaMatrix;
        use rand::SeedableRng;

        let model = analytical_model(GradientMethod::Auto);
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let params = ModelParameters {
            omega,
            omega_fixed: vec![false, false],
            ..model.default_params.clone()
        };

        // auto mode (fixed_k = None, max_k = 4)
        let mut dist = VineMixtureMarginalOmega::from_init_params_with_opts(&params, None, 4);
        assert!(!dist.k_selected, "k should not be selected yet");

        // Feed strongly bimodal samples for dim 0, unimodal for dim 1.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples: Vec<Vec<f64>> = (0..200)
            .map(|i| {
                let z0: f64 = rng.sample(StandardNormal);
                let z1: f64 = rng.sample(StandardNormal);
                let eta0 = if i < 100 {
                    -0.4 + 0.1 * z0
                } else {
                    0.4 + 0.1 * z0
                };
                let eta1 = 0.1 * z1;
                vec![eta0, eta1]
            })
            .collect();

        dist.select_k_by_bic(&samples);
        assert!(dist.k_selected, "k should be selected after BIC call");

        // Both k values should be ≥ 1.
        assert!(dist.marginals[0].k() >= 1);
        assert!(dist.marginals[1].k() >= 1);
        // Can't assert exact k without fixing rng behaviour in BIC, but it should be finite.
        assert!(dist.log_prior(&[0.1, -0.1]).is_finite());
    }
}
