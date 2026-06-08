//! Tier 2 integration smoke tests for `omega_dist = vine-multimodal`.
//!
//! These tests do not converge — they run a handful of SAEM iterations to
//! exercise the public API boundary and confirm basic contract properties
//! (Ok result, finite OFV, warnings emitted correctly). No numerical accuracy
//! is checked here; that is a Tier 3 concern.

use ferx_core::parser::model_parser::parse_model_string;
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

/// vine-multimodal with 20 subjects completes without error and returns a
/// finite OFV. Also verifies:
/// - vine_corrected_ofv is Some and finite
/// - AIC and BIC are finite
/// - AIC > OFV (penalty for mixture parameters is positive)
/// - vine_mixture_dist is populated in the result
#[test]
fn vine_multimodal_omega_dist_runs_on_simple_model() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let result = fit(&model, &population, &model.default_params, &quick_opts())
        .expect("vine-multimodal SAEM must return Ok");

    assert!(
        result.ofv.is_finite(),
        "OFV should be finite, got {}",
        result.ofv
    );
    // vine_corrected_ofv should be computed.
    assert!(
        result.vine_corrected_ofv.is_some(),
        "vine_corrected_ofv should be Some after vine-multimodal fit"
    );
    assert!(
        result.vine_corrected_ofv.unwrap().is_finite(),
        "vine_corrected_ofv should be finite"
    );

    // AIC and BIC must be finite.
    assert!(
        result.aic.is_finite(),
        "AIC should be finite, got {}",
        result.aic
    );
    assert!(
        result.bic.is_finite(),
        "BIC should be finite, got {}",
        result.bic
    );

    // AIC > OFV: mixture parameters add a positive penalty (≥ 3 params × 2 per dimension).
    assert!(
        result.aic > result.ofv,
        "AIC ({}) should be greater than OFV ({}) due to parameter penalty",
        result.aic,
        result.ofv
    );

    // vine_mixture_dist must be populated.
    assert!(
        result.vine_mixture_dist.is_some(),
        "vine_mixture_dist should be Some after vine-multimodal fit"
    );
    let mix = result.vine_mixture_dist.as_ref().unwrap();
    assert_eq!(
        mix.marginals.len(),
        2,
        "expected 2 mixture marginals (ETA_CL, ETA_V)"
    );
    for m in &mix.marginals {
        assert!(m.pi > 0.0 && m.pi < 1.0, "mixture weight must be in (0,1)");
        assert!(m.sig1 > 0.0 && m.sig2 > 0.0, "mixture SDs must be positive");
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
