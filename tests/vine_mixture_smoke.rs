//! Tier 2 integration smoke tests for `omega_dist = vine-multimodal`.
//!
//! These tests do not converge — they run a handful of SAEM iterations to
//! exercise the public API boundary and confirm basic contract properties
//! (Ok result, finite OFV, warnings emitted correctly). No numerical accuracy
//! is checked here; that is a Tier 3 concern.

use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::types::{DoseEvent, OmegaDist, Population, Subject};
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::collections::HashMap;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.16
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn make_population(n: u32) -> Population {
    let obs_times = vec![0.5_f64, 2.0, 8.0];
    let subjects = (1u32..=n)
        .map(|i| {
            let ke = 5.0 / 50.0;
            let obs: Vec<f64> = obs_times
                .iter()
                .map(|&t| (100.0 / 50.0) * (-ke * t).exp() * (1.0 + 0.02 * (i as f64 - 10.0)))
                .collect();
            Subject {
                id: format!("{i}"),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                observations: obs,
                obs_cmts: vec![1; obs_times.len()],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; obs_times.len()],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            }
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn quick_opts() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 1;
    opts.saem_n_mh_steps = 3;
    opts.saem_omega_burnin = 0;
    opts.saem_seed = Some(42);
    opts.saem_omega_dist = OmegaDist::VineMixture;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts
}

/// vine-multimodal with 20 subjects (fixed k=2) completes without error and returns:
/// - finite OFV and vine_corrected_ofv
/// - finite AIC and BIC, with AIC > OFV
/// - vine_mixture_dist populated with 2 marginals, each k=2
#[test]
fn vine_multimodal_omega_dist_runs_on_simple_model() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let result = fit(&model, &population, &model.default_params, &quick_opts())
        .expect("vine-multimodal SAEM must return Ok");

    assert!(result.ofv.is_finite(), "OFV={}", result.ofv);
    assert!(
        result.vine_corrected_ofv.is_some(),
        "vine_corrected_ofv must be Some"
    );
    assert!(
        result.vine_corrected_ofv.unwrap().is_finite(),
        "vine_corrected_ofv must be finite"
    );
    assert!(result.aic.is_finite(), "AIC={}", result.aic);
    assert!(result.bic.is_finite(), "BIC={}", result.bic);
    assert!(
        result.aic > result.ofv,
        "AIC ({}) should exceed OFV ({}) due to parameter penalty",
        result.aic,
        result.ofv
    );

    let mix = result
        .vine_mixture_dist
        .as_ref()
        .expect("vine_mixture_dist must be Some");
    assert_eq!(
        mix.marginals.len(),
        2,
        "expected 2 marginals (ETA_CL, ETA_V)"
    );
    for m in &mix.marginals {
        assert_eq!(m.k(), 2, "default fixed k=2");
        assert!(
            m.weights[0] > 0.0 && m.weights[0] < 1.0,
            "weight[0] must be in (0,1)"
        );
        assert!(m.stds[0] > 0.0 && m.stds[1] > 0.0, "SDs must be positive");
    }
}

/// With N = 10 subjects the small-N warning must be present in the result.
#[test]
fn vine_multimodal_small_n_warning_emitted() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(10);

    let result = fit(&model, &population, &model.default_params, &quick_opts())
        .expect("vine-multimodal SAEM must return Ok even for small N");

    let has_warning = result
        .warnings
        .iter()
        .any(|w| w.contains("mixture components may not be identifiable"));

    assert!(
        has_warning,
        "expected small-N warning in result.warnings, got: {:?}",
        result.warnings
    );
}

/// Fixed k=3 runs without error and reports k=3 marginals.
#[test]
fn vine_multimodal_fixed_k3_runs() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let mut opts = quick_opts();
    opts.saem_mixture_k = Some(3);

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("vine-multimodal k=3 SAEM must return Ok");

    assert!(result.ofv.is_finite(), "OFV={}", result.ofv);
    let mix = result
        .vine_mixture_dist
        .as_ref()
        .expect("vine_mixture_dist must be Some");
    for m in &mix.marginals {
        assert_eq!(m.k(), 3, "expected k=3 for all ETAs");
        assert!(
            (m.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
            "weights must sum to 1"
        );
    }
}

/// Auto k-selection (mixture_components = auto) completes without error and
/// selects k in [1, max_k] for each ETA.
#[test]
fn vine_multimodal_auto_k_runs_and_selects_valid_k() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let mut opts = quick_opts();
    opts.saem_mixture_k = None; // auto
    opts.saem_mixture_max_k = 3;
    opts.saem_omega_burnin = 2; // short burn-in so BIC selection fires

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("vine-multimodal auto-k SAEM must return Ok");

    assert!(result.ofv.is_finite(), "OFV={}", result.ofv);
    let mix = result
        .vine_mixture_dist
        .as_ref()
        .expect("vine_mixture_dist must be Some");
    for m in &mix.marginals {
        let k = m.k();
        assert!(k >= 1 && k <= 3, "selected k={k} must be in [1, max_k=3]");
    }
}

/// Parser round-trip: mixture_components = auto and max_mixture_components = 3.
#[test]
fn vine_multimodal_parser_auto_k() {
    let model_str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.16
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method                 = saem
  omega_dist             = vine-multimodal
  mixture_components     = auto
  max_mixture_components = 3
"#;
    let parsed = parse_full_model(model_str).expect("model with auto k must parse");
    assert_eq!(parsed.fit_options.saem_omega_dist, OmegaDist::VineMixture);
    assert_eq!(parsed.fit_options.saem_mixture_k, None, "None = auto");
    assert_eq!(parsed.fit_options.saem_mixture_max_k, 3);
}

/// Parser: mixture_components = 2 (fixed integer).
#[test]
fn vine_multimodal_parser_fixed_k() {
    let model_str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  omega ETA_CL ~ 0.16
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)

[structural_model]
  pk one_cpt_iv(cl=CL, v=50.0)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method             = saem
  omega_dist         = vine-multimodal
  mixture_components = 2
"#;
    let parsed = parse_full_model(model_str).expect("model with fixed k=2 must parse");
    assert_eq!(parsed.fit_options.saem_mixture_k, Some(2));
}
