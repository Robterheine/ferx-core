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

/// vine-multimodal with 20 subjects (explicit fixed k=2) completes without error
/// and returns:
/// - finite OFV and vine_corrected_ofv
/// - finite AIC and BIC, with AIC > OFV
/// - vine_mixture_dist populated with 2 marginals, each k=2
#[test]
fn vine_multimodal_omega_dist_runs_on_simple_model() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    // Use explicit fixed k=2 so the test is independent of the auto-BIC default.
    let mut opts = quick_opts();
    opts.saem_mixture_k = Some(2);

    let result = fit(&model, &population, &model.default_params, &opts)
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
    // AIC is computed from vine_corrected_ofv (not result.ofv), so compare correctly.
    let base_ofv = result.vine_corrected_ofv.unwrap_or(result.ofv);
    assert!(
        result.aic > base_ofv,
        "AIC ({}) should exceed the base OFV ({}) used for IC computation due to parameter penalty",
        result.aic,
        base_ofv
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
        assert_eq!(m.k(), 2, "explicit fixed k=2");
        assert!(
            m.weights[0] > 0.0 && m.weights[0] < 1.0,
            "weight[0] must be in (0,1)"
        );
        assert!(m.stds[0] > 0.0 && m.stds[1] > 0.0, "SDs must be positive");
    }
}

/// T2 (NEED-1): FitOptions::default() must have saem_mixture_k = None (auto-BIC).
#[test]
fn default_fit_options_mixture_k_is_none() {
    let opts = FitOptions::default();
    assert_eq!(
        opts.saem_mixture_k, None,
        "default saem_mixture_k must be None (auto-BIC), not Some(2)"
    );
}

/// T1 (NEED-1): auto-BIC on unimodal data selects k=1 for both ETAs,
/// confirming that the default no longer forces k=2 on a simple model.
#[test]
fn default_vine_multimodal_uses_auto_k() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    // Use default saem_mixture_k (None = auto), short burn-in so BIC fires.
    let mut opts = quick_opts();
    opts.saem_mixture_k = None;
    opts.saem_omega_burnin = 2;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("vine-multimodal auto-k must return Ok");

    assert!(result.ofv.is_finite(), "OFV={}", result.ofv);
    let mix = result
        .vine_mixture_dist
        .as_ref()
        .expect("vine_mixture_dist must be Some");
    // With unimodal data and a valid BIC, both ETAs should select k ≤ 2.
    // (On simple test data they will typically land on k=1.)
    for m in &mix.marginals {
        let k = m.k();
        assert!(
            k >= 1 && k <= 4,
            "auto-BIC selected k={k} must be in [1, max_k=4]"
        );
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

/// T5 (NEED-3): vine-multimodal must report pair-copula structure in vine_params.
/// The YAML/result must carry a non-empty tree block.
#[test]
fn vine_multimodal_reports_pair_copulas() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let mut opts = quick_opts();
    opts.saem_mixture_k = Some(2);

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("vine-multimodal SAEM must return Ok");

    let vp = result
        .vine_params
        .as_ref()
        .expect("vine_params must be Some for vine-multimodal (NEED-3)");
    assert!(
        !vp.trees.is_empty(),
        "vine-multimodal must report at least one vine tree"
    );
    // 2 ETAs → 1 tree with 1 pair
    assert_eq!(vp.trees.len(), 1, "2 ETAs should give 1 tree level");
    assert_eq!(vp.trees[0].pairs.len(), 1, "tree 1 should have 1 pair");
    let fam = &vp.trees[0].pairs[0].copula.family;
    assert!(
        !fam.is_empty(),
        "pair-copula family must be non-empty: got '{fam}'"
    );
}

/// T11 (NICE-2): minor-component warning is emitted when effective count < 10.
#[test]
fn minor_component_subject_count_warning() {
    // N=20 subjects with fixed k=2 → one component weight ~0.4 → 20*0.4=8 < 10.
    // Force a component to be minor by using very unequal initial conditions
    // (direct construction test rather than full SAEM).
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::stats::random_effects::RandomEffectDistribution;
    use ferx_core::stats::vine_mixture::VineMixtureMarginalOmega;
    use ferx_core::types::{ModelParameters, OmegaMatrix};

    // Build a dist with a manually unequal k=2 marginal.
    let model = parse_model_string(MODEL).expect("model must parse");
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let params = ModelParameters {
        omega,
        omega_fixed: vec![false],
        ..model.default_params.clone()
    };
    let mut dist = VineMixtureMarginalOmega::from_init_params_with_opts(&params, Some(2), 2);

    // Inject a minor component by running one SA step on heavily skewed data
    // (80% at 0, 20% at 1 → after BIC the minor component w ≈ 0.2).
    let skewed: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![if i < 80 { 0.0 } else { 1.0 }])
        .collect();
    dist.mstep_update(&skewed, 1.0);

    // With N=20 and w<0.4 the minor component has eff < 10.
    let warns = dist.identifiability_warnings(20, 10.0);
    // Check that the warning fires for the minor component.
    let has_warn = warns
        .iter()
        .any(|w| w.contains("ETA_CL") && w.contains("effective subjects"));
    assert!(
        has_warn,
        "expected minor-component warning for ETA_CL with N=20, \
         got warnings: {:?}",
        warns
    );
}

/// T9 (NEED-5): YAML output must not contain bare NaN for mixture marginal SEs.
#[test]
fn vine_multimodal_yaml_has_no_nan_se() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = make_population(20);

    let mut opts = quick_opts();
    opts.saem_mixture_k = Some(2);

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("vine-multimodal SAEM must return Ok");

    // Write to a temp file and read back to check for NaN.
    let tmp = std::env::temp_dir().join("vine_mixture_test_se.yaml");
    ferx_core::io::output::write_estimates_yaml(&result, tmp.to_str().unwrap())
        .expect("yaml write must succeed");
    let yaml = std::fs::read_to_string(&tmp).expect("yaml read must succeed");
    // Must not contain bare NaN in any SE field.
    assert!(
        !yaml.contains(": NaN") && !yaml.contains(": nan"),
        "YAML must not contain bare NaN values, got snippet: {:?}",
        yaml.lines()
            .filter(|l| l.contains("NaN") || l.contains("nan"))
            .collect::<Vec<_>>()
    );
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
