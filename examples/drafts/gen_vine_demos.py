#!/usr/bin/env python3
"""Generate two ferx-usable IV PK datasets that demonstrate the added value of
`omega_dist = vine` and `omega_dist = vine-multimodal` over a Gaussian Ω.

Scenario A  (vine)            -> data/vine_tail_iv.csv
    2-cpt IV bolus. ETA_CL and ETA_V1 share a *Clayton* copula: strong, ASYMMETRIC
    lower-tail dependence (subjects with very low clearance also have very low
    central volume), while each marginal stays Gaussian. A Gaussian Ω can only
    fit a single symmetric correlation and necessarily misses the tail asymmetry,
    so `omega_dist = vine` improves the vine-corrected OFV / AIC / BIC.

Scenario B  (vine-multimodal) -> data/vine_multimodal_iv.csv
    1-cpt IV bolus with a CYP2D6-like *bimodal* ETA_CL: ~30% poor metabolisers
    (low CL) and ~70% extensive metabolisers (high CL), ~2.7-fold apart, with no
    genotype covariate recorded. A single Gaussian marginal sits between the two
    modes and fits neither; `omega_dist = vine-multimodal` recovers k=2 for
    ETA_CL (k=1 for ETA_V) and improves OFV / AIC / BIC.

Run:  python3 examples/drafts/gen_vine_demos.py
"""
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Scenario A: 2-cpt IV bolus, Clayton lower-tail dependence between CL and V1
# ---------------------------------------------------------------------------
def two_cpt_iv_bolus(t, dose, cl, v1, q, v2):
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    disc = np.sqrt(max(s * s - 4.0 * k10 * k21, 0.0))
    alpha = 0.5 * (s + disc)
    beta = 0.5 * (s - disc)
    a = dose / v1 * (alpha - k21) / (alpha - beta)
    b = dose / v1 * (k21 - beta) / (alpha - beta)
    return a * np.exp(-alpha * t) + b * np.exp(-beta * t)


def sample_clayton(n, theta, rng):
    """Sample n pairs (u1,u2) from a Clayton copula (lower-tail dependence)."""
    u1 = rng.uniform(size=n)
    w = rng.uniform(size=n)
    u2 = (u1 ** (-theta) * (w ** (-theta / (1.0 + theta)) - 1.0) + 1.0) ** (-1.0 / theta)
    return u1, u2


def one_cpt_iv_bolus_clv(t, dose, cl, v):
    return dose / v * np.exp(-cl / v * t)


def gen_scenario_a(path, seed=20240601):
    rng = np.random.default_rng(seed)
    N = 80
    TVCL, TVV = 4.0, 50.0
    sd_cl, sd_v = 0.30, 0.30
    prop_sd = 0.10
    dose = 100.0
    times = np.array([0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48])

    # Clayton(theta=1.33): Kendall tau ~ 0.40 (MODERATE overall correlation, so the
    # Gaussian-optimal EBEs stay close to the vine optimum) but with pronounced
    # LOWER-tail dependence lambda_L = 2^(-1/theta) ~ 0.59 and zero upper-tail
    # dependence -> strongly ASYMMETRIC. A Gaussian copula (block Omega) is forced
    # to be symmetric and cannot represent this; the vine can select Clayton.
    theta = 1.33
    u_cl, u_v = sample_clayton(N, theta=theta, rng=rng)
    eta_cl = norm.ppf(u_cl) * sd_cl
    eta_v = norm.ppf(u_v) * sd_v

    rows = ["ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"]
    for i in range(N):
        cl = TVCL * np.exp(eta_cl[i])
        v = TVV * np.exp(eta_v[i])
        sid = i + 1
        rows.append(f"{sid},0,.,1,{dose:.1f},1,0,1")
        for t in times:
            cp = one_cpt_iv_bolus_clv(t, dose, cl, v)
            dv = cp * (1.0 + rng.normal(0, prop_sd))
            dv = max(dv, 1e-4)
            rows.append(f"{sid},{t:g},{dv:.4f},0,.,1,0,0")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    tau = theta / (theta + 2.0)
    lam_l = 2.0 ** (-1.0 / theta)
    print(f"[A] vine_tail_iv.csv: N={N}, Clayton theta={theta} tau~{tau:.2f} "
          f"lambdaL~{lam_l:.2f}; corr(eta_cl,eta_v)={np.corrcoef(eta_cl, eta_v)[0,1]:.2f}")


# ---------------------------------------------------------------------------
# Scenario B: 1-cpt IV bolus, bimodal ETA_CL (CYP2D6-like)
# ---------------------------------------------------------------------------
def one_cpt_iv_bolus(t, dose, cl, v):
    return dose / v * np.exp(-cl / v * t)


def gen_scenario_b(path, seed=20240602):
    rng = np.random.default_rng(seed)
    N = 80
    TVCL, TVV = 4.0, 50.0
    sd_v = 0.20
    prop_sd = 0.10
    dose = 100.0
    times = np.array([0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48])

    # Bimodal ETA_CL: weights 0.3 / 0.7, means centred so E[eta]=0, ~2.7-fold apart
    w_pm = 0.30
    m_pm, m_em = -0.70, 0.30   # 0.3*-0.7 + 0.7*0.3 = 0 ; exp(0.3 - -0.7) = e^1 ~ 2.7x
    s_pm, s_em = 0.15, 0.15

    is_pm = rng.uniform(size=N) < w_pm
    eta_cl = np.where(is_pm,
                      rng.normal(m_pm, s_pm, N),
                      rng.normal(m_em, s_em, N))
    eta_v = rng.normal(0, sd_v, N)

    rows = ["ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"]
    for i in range(N):
        cl = TVCL * np.exp(eta_cl[i])
        v = TVV * np.exp(eta_v[i])
        sid = i + 1
        rows.append(f"{sid},0,.,1,{dose:.1f},1,0,1")
        for t in times:
            cp = one_cpt_iv_bolus(t, dose, cl, v)
            dv = cp * (1.0 + rng.normal(0, prop_sd))
            dv = max(dv, 1e-4)
            rows.append(f"{sid},{t:g},{dv:.4f},0,.,1,0,0")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[B] vine_multimodal_iv.csv: N={N}, PM={int(is_pm.sum())} "
          f"({is_pm.mean()*100:.0f}%), EM={int((~is_pm).sum())}; "
          f"mode separation exp({m_em-m_pm:.1f})={np.exp(m_em-m_pm):.1f}-fold")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(here, "..", "..", "data"))
    gen_scenario_a(os.path.join(data_dir, "vine_tail_iv.csv"))
    gen_scenario_b(os.path.join(data_dir, "vine_multimodal_iv.csv"))
    print("done.")
