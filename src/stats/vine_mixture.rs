//! Vine copula with 2-component Gaussian mixture marginals.
//!
//! `VineMixtureMarginalOmega` is the estimator for `omega_dist = vine-multimodal`.
//! It replaces the per-ETA Gaussian marginal in [`VineCopulaOmega`] with a
//! [`MixtureMarginal`], feeding the resulting PIT values into the same D-vine
//! pair-copula machinery.
//!
//! This captures bimodal marginal distributions (e.g. CYP2D6 poor-metaboliser
//! sub-populations) while retaining flexible tail dependence between ETAs.
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

/// Minimum and maximum mixing weight — keeps both components alive.
const MIXING_FLOOR: f64 = 0.05;

/// Minimum marginal standard deviation for the Gaussian-equivalent OMEGA floor.
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

// ---------------------------------------------------------------------------
// MixtureMarginal
// ---------------------------------------------------------------------------

/// 2-component Gaussian mixture marginal for one ETA dimension.
///
/// The density is:
/// ```text
/// f(x) = π · φ(x; μ₁, σ₁) + (1−π) · φ(x; μ₂, σ₂)
/// ```
/// and the CDF is:
/// ```text
/// F(x) = π · Φ((x − μ₁)/σ₁) + (1−π) · Φ((x − μ₂)/σ₂)
/// ```
///
/// Identifiability constraint: `μ₁ ≤ μ₂` is enforced at construction time
/// and after every EM M-step by swapping components when violated.
#[derive(Clone, Debug)]
pub struct MixtureMarginal {
    /// Mixing weight of component 1 (component 2 weight = 1 − π).
    pub pi: f64,
    /// Mean of component 1 (≤ μ₂).
    pub mu1: f64,
    /// Standard deviation of component 1 (≥ MARGINAL_STD_FLOOR).
    pub sig1: f64,
    /// Mean of component 2 (≥ μ₁).
    pub mu2: f64,
    /// Standard deviation of component 2 (≥ MARGINAL_STD_FLOOR).
    pub sig2: f64,
}

/// Standard normal PDF φ(z) = exp(−z²/2) / √(2π).
#[inline]
fn standard_normal_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

impl MixtureMarginal {
    /// Construct from raw parameters, enforcing constraints.
    pub fn new(pi: f64, mu1: f64, sig1: f64, mu2: f64, sig2: f64) -> Self {
        let mut m = Self {
            pi: pi.clamp(MIXING_FLOOR, 1.0 - MIXING_FLOOR),
            mu1,
            sig1: sig1.max(MARGINAL_STD_FLOOR),
            mu2,
            sig2: sig2.max(MARGINAL_STD_FLOOR),
        };
        m.enforce_label_order();
        m
    }

    /// Ensure μ₁ ≤ μ₂ by swapping components when the constraint is violated.
    fn enforce_label_order(&mut self) {
        if self.mu1 > self.mu2 {
            std::mem::swap(&mut self.mu1, &mut self.mu2);
            std::mem::swap(&mut self.sig1, &mut self.sig2);
            self.pi = 1.0 - self.pi;
        }
    }

    /// Mixture PDF at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        self.pi * standard_normal_pdf((x - self.mu1) / self.sig1) / self.sig1
            + (1.0 - self.pi) * standard_normal_pdf((x - self.mu2) / self.sig2) / self.sig2
    }

    /// Mixture CDF at `x` (= the PIT value u = F(x)).
    pub fn cdf(&self, x: f64) -> f64 {
        self.pi * normal_cdf((x - self.mu1) / self.sig1)
            + (1.0 - self.pi) * normal_cdf((x - self.mu2) / self.sig2)
    }

    /// Log-density: log f(x). Returns a large negative value for effectively-zero density.
    pub fn log_pdf(&self, x: f64) -> f64 {
        let p = self.pdf(x);
        if p > 0.0 {
            p.ln()
        } else {
            -1e20
        }
    }

    /// Inverse CDF via bisection on the bracketing interval
    /// `[μ_overall − 8·σ_overall, μ_overall + 8·σ_overall]`.
    ///
    /// Tolerance: 1e-9. Always converges in ≤ 60 iterations.
    pub fn icdf(&self, u: f64) -> f64 {
        let u = u.clamp(1e-12, 1.0 - 1e-12);

        // Overall marginal mean and SD for the search window.
        let mu = self.pi * self.mu1 + (1.0 - self.pi) * self.mu2;
        let var = self.pi * (self.sig1 * self.sig1 + self.mu1 * self.mu1)
            + (1.0 - self.pi) * (self.sig2 * self.sig2 + self.mu2 * self.mu2)
            - mu * mu;
        let sigma = var.max(0.0).sqrt().max(MARGINAL_STD_FLOOR);

        let mut lo = mu - 8.0 * sigma;
        let mut hi = mu + 8.0 * sigma;

        // Guarantee the bracket straddles u.
        // Expand if cdf(lo) > u or cdf(hi) < u (shouldn't happen for 8σ window).
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

    /// Fit by EM from a slice of ETA samples.
    ///
    /// Initialisation: component 1 takes samples below the median, component 2
    /// takes samples above. This splits-at-median initialisation avoids the
    /// degenerate local mode where both components are identical.
    ///
    /// EM converges when the parameter change is < 1e-6 or after 100 iterations.
    /// Constraints applied after every M-step:
    /// - `π ∈ [MIXING_FLOOR, 1 − MIXING_FLOOR]`
    /// - `σ₁, σ₂ ≥ MARGINAL_STD_FLOOR`
    /// - `μ₁ ≤ μ₂` (label ordering)
    pub fn fit_em(samples: &[f64]) -> Self {
        let n = samples.len();
        if n < 4 {
            // Fallback: single-component near zero.
            let mean = if n > 0 {
                samples.iter().sum::<f64>() / n as f64
            } else {
                0.0
            };
            let sd = if n > 1 {
                let var =
                    samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
                var.sqrt().max(MARGINAL_STD_FLOOR)
            } else {
                MARGINAL_STD_FLOOR
            };
            return MixtureMarginal::new(0.5, mean - 0.5 * sd, sd, mean + 0.5 * sd, sd);
        }

        // Initialise: split at median.
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[n / 2];

        let comp1: Vec<f64> = samples.iter().copied().filter(|&x| x <= median).collect();
        let comp2: Vec<f64> = samples.iter().copied().filter(|&x| x > median).collect();

        let init_mean_sd = |v: &[f64]| -> (f64, f64) {
            if v.is_empty() {
                return (0.0, MARGINAL_STD_FLOOR);
            }
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s = if v.len() > 1 {
                let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0);
                var.sqrt().max(MARGINAL_STD_FLOOR)
            } else {
                MARGINAL_STD_FLOOR
            };
            (m, s)
        };

        let (m1, s1) = init_mean_sd(&comp1);
        let (m2, s2) = init_mean_sd(&comp2);
        let init_pi = (comp1.len() as f64 / n as f64).clamp(MIXING_FLOOR, 1.0 - MIXING_FLOOR);

        let mut pi = init_pi;
        let mut mu1 = m1;
        let mut sig1 = s1;
        let mut mu2 = m2;
        let mut sig2 = s2;

        let mut responsibilities = vec![0.0f64; n];

        for _iter in 0..100 {
            // E-step: compute responsibility r_i = π·φ₁(xᵢ) / (π·φ₁(xᵢ) + (1−π)·φ₂(xᵢ)).
            for (i, &x) in samples.iter().enumerate() {
                let p1 = pi * standard_normal_pdf((x - mu1) / sig1) / sig1;
                let p2 = (1.0 - pi) * standard_normal_pdf((x - mu2) / sig2) / sig2;
                let denom = p1 + p2;
                responsibilities[i] = if denom > 0.0 { p1 / denom } else { 0.5 };
            }

            // M-step.
            let n1: f64 = responsibilities.iter().sum();
            let n2 = n as f64 - n1;

            let new_pi = (n1 / n as f64).clamp(MIXING_FLOOR, 1.0 - MIXING_FLOOR);

            let new_mu1 = if n1 > 0.0 {
                responsibilities
                    .iter()
                    .zip(samples)
                    .map(|(r, x)| r * x)
                    .sum::<f64>()
                    / n1
            } else {
                mu1
            };
            let new_mu2 = if n2 > 0.0 {
                responsibilities
                    .iter()
                    .zip(samples)
                    .map(|(r, x)| (1.0 - r) * x)
                    .sum::<f64>()
                    / n2
            } else {
                mu2
            };

            let new_sig1 = if n1 > 1.0 {
                (responsibilities
                    .iter()
                    .zip(samples)
                    .map(|(r, x)| r * (x - new_mu1) * (x - new_mu1))
                    .sum::<f64>()
                    / n1)
                    .sqrt()
                    .max(MARGINAL_STD_FLOOR)
            } else {
                MARGINAL_STD_FLOOR
            };
            let new_sig2 = if n2 > 1.0 {
                (responsibilities
                    .iter()
                    .zip(samples)
                    .map(|(r, x)| (1.0 - r) * (x - new_mu2) * (x - new_mu2))
                    .sum::<f64>()
                    / n2)
                    .sqrt()
                    .max(MARGINAL_STD_FLOOR)
            } else {
                MARGINAL_STD_FLOOR
            };

            // Check convergence.
            let delta = (new_pi - pi).abs()
                + (new_mu1 - mu1).abs()
                + (new_sig1 - sig1).abs()
                + (new_mu2 - mu2).abs()
                + (new_sig2 - sig2).abs();

            pi = new_pi;
            mu1 = new_mu1;
            sig1 = new_sig1;
            mu2 = new_mu2;
            sig2 = new_sig2;

            if delta < 1e-6 {
                break;
            }
        }

        MixtureMarginal::new(pi, mu1, sig1, mu2, sig2)
    }

    /// Overall marginal mean E[X].
    pub fn mean(&self) -> f64 {
        self.pi * self.mu1 + (1.0 - self.pi) * self.mu2
    }

    /// Overall marginal standard deviation √Var[X].
    pub fn std_dev(&self) -> f64 {
        let mu = self.mean();
        let var = self.pi * (self.sig1 * self.sig1 + self.mu1 * self.mu1)
            + (1.0 - self.pi) * (self.sig2 * self.sig2 + self.mu2 * self.mu2)
            - mu * mu;
        var.max(0.0).sqrt().max(MARGINAL_STD_FLOOR)
    }
}

// ---------------------------------------------------------------------------
// VineMixtureMarginalOmega
// ---------------------------------------------------------------------------

/// Vine copula with per-ETA 2-component Gaussian mixture marginals.
///
/// This implements `omega_dist = vine-multimodal`. Each η dimension gets an
/// independent [`MixtureMarginal`]; the PIT values `u_i = F_i(η_i)` are fed
/// into the same D-vine copula structure used by [`VineCopulaOmega`].
///
/// The estimator follows a three-step M-step pattern:
/// 1. **Mixture marginals**: re-fit each [`MixtureMarginal`] by EM from the
///    current pooled SAEM η samples.
/// 2. **Gaussian-equivalent OMEGA**: update the sample covariance for reporting
///    and the MH proposal scale, using the same SA-averaging as the Gaussian arm.
/// 3. **Pair-copulas**: fit/re-fit the D-vine pair-copulas from PIT pseudo-observations.
#[derive(Debug, Clone)]
pub struct VineMixtureMarginalOmega {
    /// Number of ETA dimensions.
    pub d: usize,
    /// Per-dimension 2-component Gaussian mixture marginals.
    /// Index `i` corresponds to the variable at `variable_order[i]` in the original η.
    pub marginals: Vec<MixtureMarginal>,
    /// D-vine pair-copula families. `pair_copulas[k][j]` is at tree k+1, pair j.
    pub pair_copulas: Vec<Vec<CopulaFamily>>,
    /// Whether AIC-based family selection has run at least once.
    families_selected: bool,
    /// SA sufficient statistic s₂ = (1−γ) s₂_prev + γ (1/N) Σ ηᵢηᵢᵀ.
    /// Maintained in original η coordinates (not vine-reordered).
    sample_s2: DMatrix<f64>,
    /// Gaussian-equivalent OmegaMatrix from `sample_s2`.
    /// Used for MH proposal scale and as the OMEGA output in reports.
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
    /// Currently natural order (0, 1, …, d−1); reserved for future optimisation.
    pub variable_order: Vec<usize>,
}

impl VineMixtureMarginalOmega {
    /// Construct from the initial model parameters.
    ///
    /// Each marginal is initialised as a symmetric 2-component mixture centred
    /// on 0 with components at ±0.5·σ. This deliberately diffuse initialisation
    /// lets EM find the true components once SAEM samples accumulate.
    pub fn from_init_params(init_params: &ModelParameters) -> Self {
        let omega = init_params.omega.clone();
        let d = omega.dim();

        // Variable order: natural (identity permutation) for now.
        let variable_order: Vec<usize> = (0..d).collect();

        // Symmetric two-component initialisation.
        let marginals: Vec<MixtureMarginal> = (0..d)
            .map(|i| {
                let sigma = omega.matrix[(i, i)]
                    .max(MARGINAL_STD_FLOOR * MARGINAL_STD_FLOOR)
                    .sqrt();
                MixtureMarginal::new(0.5, -0.5 * sigma, sigma, 0.5 * sigma, sigma)
            })
            .collect();

        // Default: independent Gaussian copulas (ρ=0) before the first M-step.
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
            sample_s2,
            omega_equiv,
            initial_omega,
            omega_fixed,
            initial_matrix,
            per_pair_pseudo_obs: Vec::new(),
            variable_order,
        }
    }

    /// Compute PIT pseudo-observations from η samples using the current mixture marginals.
    ///
    /// Returns `u[j][vine_pos] = F_{vine_pos}(η[j][orig_pos])` clamped to (1e-8, 1−1e-8).
    pub(crate) fn pit_from_samples(&self, sampled_etas: &[Vec<f64>]) -> Vec<Vec<f64>> {
        sampled_etas
            .iter()
            .map(|eta| {
                self.variable_order
                    .iter()
                    .enumerate()
                    .map(|(vine_pos, &orig)| {
                        let _ = vine_pos; // vine_pos == index in result vec
                        self.marginals[orig].cdf(eta[orig]).clamp(1e-8, 1.0 - 1e-8)
                    })
                    .collect()
            })
            .collect()
    }

    /// Re-fit each mixture marginal by EM from the per-dimension samples.
    fn update_marginals_em(&mut self, sampled_etas: &[Vec<f64>]) {
        for i in 0..self.d {
            let col: Vec<f64> = sampled_etas.iter().map(|e| e[i]).collect();
            self.marginals[i] = MixtureMarginal::fit_em(&col);
        }
    }

    /// Update the Gaussian-equivalent OMEGA via SA averaging (mirrors VineCopulaOmega).
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
    ///
    /// Mirrors `VineCopulaOmega::fit_vine` with the same damped-parameter
    /// update during exploration and AIC-based family selection on the first call.
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
    /// Negative log joint density: −log p(η) = Σᵢ [−log fᵢ(ηᵢ)] − log c(u₁,…,u_d).
    fn log_prior(&self, eta: &[f64]) -> f64 {
        let mut log_prior = 0.0;
        let mut u = vec![0.0f64; self.d];

        for (vine_pos, &orig) in self.variable_order.iter().enumerate() {
            // Negative log marginal density.
            log_prior -= self.marginals[orig].log_pdf(eta[orig]);
            // PIT value for copula.
            u[vine_pos] = self.marginals[orig]
                .cdf(eta[orig])
                .clamp(1e-12, 1.0 - 1e-12);
        }

        // Subtract vine log-density.
        if self.d > 1 {
            let vine_ld = dvine_log_density(&u, &self.pair_copulas);
            log_prior -= vine_ld;
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
        // 1. Re-fit mixture marginals by EM.
        self.update_marginals_em(sampled_etas);
        // 2. Update Gaussian-equivalent OMEGA for reporting and MH proposal.
        self.update_sample_cov(sampled_etas, gamma);
        // 3. Re-fit pair-copulas from mixture PIT pseudo-observations.
        let pseudo_obs = self.pit_from_samples(sampled_etas);
        let select = !self.families_selected;
        self.fit_vine(&pseudo_obs, select, gamma);
        if select {
            self.families_selected = true;
        }
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        // Sample from the Gaussian-equivalent Ω for reporting.
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
    ///
    /// Mirrors `VineCopulaOmega::draw_eta` exactly, with the final marginal
    /// inversion replaced by `marginals[k].icdf(u)` instead of the Gaussian
    /// `μ + σ · Φ⁻¹(u)`.
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

        // V-table: same structure as VineCopulaOmega::draw_eta.
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

        // Invert mixture marginals: u[k] → η[k] = F_{orig_k}⁻¹(u[k]).
        // Result is in vine order; map back to original η order.
        let mut result = vec![0.0f64; d];
        for k in 0..d {
            let orig = self.variable_order[k];
            result[orig] = self.marginals[orig].icdf(vt[k][k]);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Standard errors for mixture marginal parameters (Rung 4 placeholder)
// ---------------------------------------------------------------------------

/// Approximate standard errors for the 5 mixture marginal parameters
/// (π, μ₁, σ₁, μ₂, σ₂) from the observed Fisher information matrix.
///
/// Currently unimplemented — returns `[f64::NAN; 5]`. Will be replaced in
/// Rung 4 with FD-based Hessian inversion on the per-subject ETA samples.
pub fn mixture_marginal_se(_m: &MixtureMarginal, _samples: &[f64]) -> [f64; 5] {
    [f64::NAN; 5]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, ModelParameters};

    // ── MixtureMarginal tests ────────────────────────────────────────────────

    /// CDF must be non-decreasing on a grid.
    #[test]
    fn mixture_marginal_cdf_monotone() {
        let m = MixtureMarginal::new(0.6, -0.5, 0.2, 0.5, 0.2);
        let grid: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.2).collect();
        let mut prev = m.cdf(grid[0]);
        for &x in &grid[1..] {
            let cur = m.cdf(x);
            assert!(
                cur >= prev - 1e-12,
                "CDF not monotone: cdf({x:.1})={cur:.6} < cdf(prev)={prev:.6}"
            );
            prev = cur;
        }
    }

    /// icdf must round-trip: icdf(cdf(x)) ≈ x for a range of x.
    #[test]
    fn mixture_marginal_icdf_roundtrip() {
        let m = MixtureMarginal::new(0.6, -0.5, 0.15, 0.5, 0.15);
        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let u = m.cdf(x);
            let x_back = m.icdf(u);
            assert!(
                (x_back - x).abs() < 1e-6,
                "icdf(cdf({x}))={x_back:.9} expected ≈ {x}"
            );
        }
    }

    /// EM should recover a known bimodal mixture from 1 000 samples.
    #[test]
    fn mixture_em_recovers_two_components() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // True mixture: 60% N(−0.5, 0.1²), 40% N(0.5, 0.1²).
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

        let m = MixtureMarginal::fit_em(&samples);

        // Components must have mu1 < mu2 (label ordering).
        assert!(m.mu1 < m.mu2, "mu1={:.3} must be < mu2={:.3}", m.mu1, m.mu2);
        // mu1 ≈ −0.5, mu2 ≈ 0.5.
        assert!(
            (m.mu1 - (-0.5)).abs() < 0.1,
            "mu1={:.3} expected ≈ -0.5",
            m.mu1
        );
        assert!((m.mu2 - 0.5).abs() < 0.1, "mu2={:.3} expected ≈ 0.5", m.mu2);
        // pi ≈ 0.6 (the heavier component).
        assert!((m.pi - 0.6).abs() < 0.05, "pi={:.3} expected ≈ 0.6", m.pi);
    }

    /// Label ordering: initialising with mu1 > mu2 must be swapped by new().
    #[test]
    fn mixture_label_switch_enforced() {
        let m = MixtureMarginal::new(0.3, 1.0, 0.1, -1.0, 0.1);
        assert!(
            m.mu1 <= m.mu2,
            "label order not enforced: mu1={:.3} mu2={:.3}",
            m.mu1,
            m.mu2
        );
    }

    // ── VineMixtureMarginalOmega tests ───────────────────────────────────────

    /// log_prior must be finite for a 1-ETA model at several η values.
    #[test]
    fn vine_mixture_log_prior_finite() {
        let model = analytical_model(GradientMethod::Auto);
        let dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        for &eta in &[0.0, 0.5, -0.3, 1.2] {
            let lp = dist.log_prior(&[eta]);
            assert!(lp.is_finite(), "log_prior([{eta}]) = {lp} is not finite");
        }
    }

    /// variable_order must be a permutation of 0..d.
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

    /// log_prior values should change after mstep_update with informative samples.
    #[test]
    fn vine_mixture_mstep_updates_marginals() {
        let model = analytical_model(GradientMethod::Auto);
        let mut dist = VineMixtureMarginalOmega::from_init_params(&model.default_params);
        let lp_before = dist.log_prior(&[0.3]);

        // Feed bimodal samples to drive the marginal away from the symmetric init.
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

        // After update the prior value should be finite and different.
        assert!(lp_after.is_finite(), "log_prior after mstep = {lp_after}");
        assert_ne!(
            lp_before, lp_after,
            "log_prior unchanged after mstep_update"
        );
    }

    /// draw_eta returns a vector of the right length with finite entries.
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

    /// 2-ETA model: log_prior is finite and draw_eta returns length-2 vectors.
    #[test]
    fn vine_mixture_two_eta_log_prior_and_draw() {
        use crate::types::{OmegaMatrix, SigmaVector};

        // Build a 2-ETA diagonal ModelParameters.
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
        assert_eq!(dist.variable_order, vec![0, 1]);

        let lp = dist.log_prior(&[0.1, -0.2]);
        assert!(lp.is_finite(), "2-ETA log_prior = {lp}");

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let eta = dist.draw_eta(&mut rng);
        assert_eq!(eta.len(), 2);
        assert!(eta.iter().all(|x| x.is_finite()));
    }
}
