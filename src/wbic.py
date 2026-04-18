"""
WBIC (Watanabe-Bayesian Information Criterion) fitting for the two-step task.

Each model is fit per-session using a tempered posterior with
    beta = 1 / log(T)   (T = number of trials in the session)

WBIC = -2 * E_beta[ log p(y | theta) ]
     = -2 * mean over posterior samples of log_lik

Models
------
  mf_p      : Model-free + perseveration          (alpha, forget, lambda, Wmf, P)
  rac_p     : Reward-as-cue + perseveration       (alpha, tmp, P)
  hyb_p     : Hybrid (MF+MB) + perseveration      (alpha, forget, lambda, Wmf, Wmb, P)
  ls        : Latent State (Akam 2015)             (p_r, beta)
  ls_asym_p : Latent State Asymmetric + P          (p_r, i_temp, P)
  hyb_inf   : Hybrid (Asym Inference + MF) + P    (alpha, forget, lambda, Wmf, Winf, P, p_r)

Usage
-----
    from src.wbic import WBIC_MODELS, compile_wbic_models, fit_wbic_session
    from src.fitting import RL_data_arrange_single

    compiled = compile_wbic_models(['mf_p', 'ls'])

    T, c, ss, tt, r, PR, PL = RL_data_arrange_single(session)
    result = fit_wbic_session(compiled['mf_p'], WBIC_MODELS['mf_p'],
                               T, c, ss, tt, r, PR, PL)
    # result: {'WBIC': float, 'alpha': float, ...}
"""
from collections import namedtuple
import numpy as np

WBICModel = namedtuple('WBICModel', ['stan_code', 'param_names', 'r_is_int'])


# ============================================================================
# Per-session WBIC Stan codes
# Each model block is duplicated in generated quantities to compute log_lik.
# Differences from the multi-session registry Stan codes:
#   - Single-session data block (T, c[T], ss[T], tt[T], r[T], PR[2], PL[2])
#   - transformed data: watanabe_beta = 1.0 / log(T)
#   - target += watanabe_beta * log(p_t)   (tempered likelihood in model block)
#   - generated quantities: log_lik = sum(log p_t)  (untempered; WBIC = -mean(log_lik))
# ============================================================================

# ----------------------------------------------------------------------------
# 1. mf_p  —  Model-free + single perseveration
# ----------------------------------------------------------------------------
_STAN_WBIC_MF_P = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  real r[T];
}
transformed data {
  real<lower=0, upper=1> watanabe_beta; // WBIC parameter
  watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> forget;
  real<lower=0, upper=1> lambda;
  real<lower=0> Wmf;
  real P;
}
model {
  alpha  ~ beta(1.2, 1.2);
  forget ~ beta(1.2, 1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf    ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);
  {
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    real prev_c = 0.0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t]   + alpha*(lambda*r[t] - (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        V[ss[t],    t+1] = (1-alpha)*V[ss[t],t]    + alpha*r[t];
        V[3-ss[t],  t+1] = (1-forget)*V[3-ss[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qnet[1, t+1] = Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmf*Qmf[2,t+1];
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    real prev_c = 0.0;
    real ll = 0.0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t]   + alpha*(lambda*r[t] + (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        V[ss[t],    t+1] = (1-alpha)*V[ss[t],t]    + alpha*r[t];
        V[3-ss[t],  t+1] = (1-forget)*V[3-ss[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qnet[1, t+1] = Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmf*Qmf[2,t+1];
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 2. rac_p  —  Reward-as-cue + single perseveration
# ----------------------------------------------------------------------------
_STAN_WBIC_RAC_P = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  int r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0> tmp;
  real P;
}
model {
  alpha ~ beta(1.2, 1.2);
  tmp   ~ gamma(2, 0.4);
  P     ~ student_t(4, 0, 2.5);
  {
    vector[2] Q;
    real Q_td[2, 2, 2];
    real prev_c = 0.0;
    int  ps;
    int  po;
    int  c_int;
    Q[1] = 0.5; Q[2] = 0.5;
    for (a in 1:2) for (s in 1:2) for (o in 1:2) Q_td[a,s,o] = 0.5;
    ps = 1; po = 1;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-tmp * (Q[c[t]] - Q[3-c[t]]))));
      if (t < T) {
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        c_int  = c[t];
        Q_td[c_int, ps, po] = (1.0 - alpha) * Q_td[c_int, ps, po] + alpha * r[t];
        ps = ss[t];
        po = r[t] + 1;
        Q[1] = Q_td[1, ps, po] + P * prev_c;
        Q[2] = Q_td[2, ps, po];
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    vector[2] Q;
    real Q_td[2, 2, 2];
    real prev_c = 0.0;
    int  ps;
    int  po;
    int  c_int;
    real ll = 0.0;
    Q[1] = 0.5; Q[2] = 0.5;
    for (a in 1:2) for (s in 1:2) for (o in 1:2) Q_td[a,s,o] = 0.5;
    ps = 1; po = 1;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-tmp * (Q[c[t]] - Q[3-c[t]]))));
      if (t < T) {
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        c_int  = c[t];
        Q_td[c_int, ps, po] = (1.0 - alpha) * Q_td[c_int, ps, po] + alpha * r[t];
        ps = ss[t];
        po = r[t] + 1;
        Q[1] = Q_td[1, ps, po] + P * prev_c;
        Q[2] = Q_td[2, ps, po];
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 3. hyb_p  —  Hybrid (MF + MB) + single perseveration
# ----------------------------------------------------------------------------
_STAN_WBIC_HYB_P = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  real r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> forget;
  real<lower=0, upper=1> lambda;
  real<lower=0> Wmf;
  real<lower=0> Wmb;
  real P;
}
model {
  alpha  ~ beta(1.2, 1.2);
  forget ~ beta(1.2, 1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf    ~ gamma(2, 0.4);
  Wmb    ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);
  {
    matrix[2,T] Qmb;
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    real prev_c = 0.0;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t]   + alpha*(lambda*r[t] + (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        V[ss[t],    t+1] = (1-alpha)*V[ss[t],t]    + alpha*r[t];
        V[3-ss[t],  t+1] = (1-forget)*V[3-ss[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qmb[1, t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2, t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        Qnet[1, t+1] = Wmb*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmb*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    matrix[2,T] Qmb;
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    real prev_c = 0.0;
    real ll = 0.0;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t]   + alpha*(lambda*r[t] + (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        V[ss[t],    t+1] = (1-alpha)*V[ss[t],t]    + alpha*r[t];
        V[3-ss[t],  t+1] = (1-forget)*V[3-ss[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qmb[1, t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2, t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        Qnet[1, t+1] = Wmb*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmb*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 4. mb_p  —  Model-based + single perseveration
# ----------------------------------------------------------------------------
_STAN_WBIC_MB_P = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  real r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> forget;
  real<lower=0> Wmb;
  real P;
}
model {
  alpha  ~ beta(1.2, 1.2);
  forget ~ beta(1.2, 1.2);
  Wmb    ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);
  {
    matrix[2,T] V;
    matrix[2,T] Qmb;
    matrix[2,T] Qnet;
    real prev_c = 0.0;
    V[1,1]    = 0; V[2,1]    = 0;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        V[ss[t],   t+1] = (1-alpha)*V[ss[t],t]   + alpha*r[t];
        V[3-ss[t], t+1] = (1-forget)*V[3-ss[t],t];
        Qmb[1, t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2, t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qnet[1, t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmb*Qmb[2,t+1];
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    matrix[2,T] V;
    matrix[2,T] Qmb;
    matrix[2,T] Qnet;
    real prev_c = 0.0;
    real ll = 0.0;
    V[1,1]    = 0; V[2,1]    = 0;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        V[ss[t],   t+1] = (1-alpha)*V[ss[t],t]   + alpha*r[t];
        V[3-ss[t], t+1] = (1-forget)*V[3-ss[t],t];
        Qmb[1, t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2, t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        Qnet[1, t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2, t+1] = Wmb*Qmb[2,t+1];
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 5. ls  —  Latent State (Akam 2015)
# Akam 2015 (reduced task) — rewritten 2026-04-17, smoothed 2026-04-18.
# Bayesian inference on (ss, r) using reward probabilities Pgood=0.8, Pbad=0.2.
# Choice rule: posterior-weighted ε-greedy (probability-matching over latent
# state) — p_agree = p_1 if c == best_a_S1 else (1 - p_1), then
# p_c = p_agree*(1-eps) + (1-p_agree)*eps. Matches latent_state._STAN_LS.
_STAN_WBIC_LS = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  real r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
  real PGOOD = 0.8;
  real PBAD  = 0.2;
  int best_a_S1;  // best action in S1 (max P(ss=1 | a))
  int best_a_S2;
  if (PL[1] > PR[1]) { best_a_S1 = 1; best_a_S2 = 2; }
  else               { best_a_S1 = 2; best_a_S2 = 1; }
}
parameters {
  real<lower=0, upper=1>   p_r;  // block reversal probability
  real<lower=0, upper=0.5> eps;  // ε for ε-greedy
}
model {
  p_r ~ beta(2, 2);
  eps ~ beta(1.2, 1.2);
  {
    real p_1[T];
    real lik_S1;
    real lik_S2;
    real post;
    real p_c;
    real p_agree;
    p_1[1] = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1) {
        if (c[t] == best_a_S1) p_agree = p_1[t];
        else                   p_agree = 1 - p_1[t];
        p_c = p_agree * (1 - eps) + (1 - p_agree) * eps;
        target += watanabe_beta * log(p_c + 1e-16);
      }
      if (t < T) {
        if      (ss[t]==1 && r[t]==1) { lik_S1 = PGOOD;   lik_S2 = PBAD;   }
        else if (ss[t]==2 && r[t]==1) { lik_S1 = PBAD;    lik_S2 = PGOOD;  }
        else if (ss[t]==1 && r[t]==0) { lik_S1 = 1-PGOOD; lik_S2 = 1-PBAD; }
        else                          { lik_S1 = 1-PBAD;  lik_S2 = 1-PGOOD;}
        post = lik_S1 * p_1[t] / (lik_S1 * p_1[t] + lik_S2 * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * post + p_r * (1 - post);
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    real p_1[T];
    real lik_S1;
    real lik_S2;
    real post;
    real p_c;
    real p_agree;
    real ll = 0.0;
    p_1[1] = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1) {
        if (c[t] == best_a_S1) p_agree = p_1[t];
        else                   p_agree = 1 - p_1[t];
        p_c = p_agree * (1 - eps) + (1 - p_agree) * eps;
        ll += log(p_c + 1e-16);
      }
      if (t < T) {
        if      (ss[t]==1 && r[t]==1) { lik_S1 = PGOOD;   lik_S2 = PBAD;   }
        else if (ss[t]==2 && r[t]==1) { lik_S1 = PBAD;    lik_S2 = PGOOD;  }
        else if (ss[t]==1 && r[t]==0) { lik_S1 = 1-PGOOD; lik_S2 = 1-PBAD; }
        else                          { lik_S1 = 1-PBAD;  lik_S2 = 1-PGOOD;}
        post = lik_S1 * p_1[t] / (lik_S1 * p_1[t] + lik_S2 * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * post + p_r * (1 - post);
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 5. ls_asym_p  —  Latent State Asymmetric + single perseveration
# Note: fixes typo 'fs_int' → 's_int' in the multi-session Stan code.
# ----------------------------------------------------------------------------
_STAN_WBIC_LS_ASYM_P = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  int r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> p_r;
  real<lower=0>          i_temp;
  real                   P;
}
model {
  p_r    ~ beta(1.2, 1.2);
  i_temp ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);
  {
    matrix[2,T] V;
    matrix[2,T] Qmb;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c = 0.0;
    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;
    V[1,1] = 0.5; V[2,1] = 0.5;
    Qmb[1,1] = 0.5; Qmb[2,1] = 0.5;
    p_1[1] = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-(Qmb[c[t],t] - Qmb[3-c[t],t]))));
      if (t < T) {
        s_int  = ss[t];
        o_int  = r[t] + 1;
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        p_1[t+1] = p_o_1[s_int,o_int] * p_1[t] /
                   (p_o_1[s_int,o_int] * p_1[t] + p_o_0[s_int,o_int] * (1 - p_1[t]));
        p_1[t+1] = (1-p_r)*p_1[t+1] + p_r*(1-p_1[t+1]);
        V1 = 0.8*p_1[t+1] + 0.2*(1-p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;
        Qmb[1,t+1] = i_temp*(PL[1]*V[1,t+1] + PL[2]*V[2,t+1]) + P*prev_c;
        Qmb[2,t+1] = i_temp*(PR[1]*V[1,t+1] + PR[2]*V[2,t+1]);
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    matrix[2,T] V;
    matrix[2,T] Qmb;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c = 0.0;
    real ll = 0.0;
    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;
    V[1,1] = 0.5; V[2,1] = 0.5;
    Qmb[1,1] = 0.5; Qmb[2,1] = 0.5;
    p_1[1] = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-(Qmb[c[t],t] - Qmb[3-c[t],t]))));
      if (t < T) {
        s_int  = ss[t];
        o_int  = r[t] + 1;
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        p_1[t+1] = p_o_1[s_int,o_int] * p_1[t] /
                   (p_o_1[s_int,o_int] * p_1[t] + p_o_0[s_int,o_int] * (1 - p_1[t]));
        p_1[t+1] = (1-p_r)*p_1[t+1] + p_r*(1-p_1[t+1]);
        V1 = 0.8*p_1[t+1] + 0.2*(1-p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;
        Qmb[1,t+1] = i_temp*(PL[1]*V[1,t+1] + PL[2]*V[2,t+1]) + P*prev_c;
        Qmb[2,t+1] = i_temp*(PR[1]*V[1,t+1] + PR[2]*V[2,t+1]);
      }
    }
    log_lik = ll;
  }
}
"""

# ----------------------------------------------------------------------------
# 6. hyb_inf  —  Hybrid (Asymmetric Inference + MF) + single perseveration
# Note: uses 'Winf' consistently (matching param_names), fixing the Wmb/Winf
#       naming inconsistency in the original STAN_HYB_INF.
# ----------------------------------------------------------------------------
_STAN_WBIC_HYB_INF = """
data {
  int<lower=1> T;
  real PR[2];
  real PL[2];
  int<lower=0,upper=2> c[T];
  int<lower=0,upper=2> ss[T];
  int<lower=0,upper=2> tt[T];
  int r[T];
}
transformed data {
  real watanabe_beta = 1.0 / log(T);
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> forget;
  real<lower=0, upper=1> lambda;
  real<lower=0>          Wmf;
  real<lower=0>          Winf;
  real<lower=0>          P;
  real<lower=0, upper=1> p_r;
}
model {
  alpha  ~ beta(1.2, 1.2);
  forget ~ beta(1.2, 1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf    ~ gamma(2, 0.4);
  Winf   ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);
  p_r    ~ beta(1.2, 1.2);
  {
    matrix[2,T] Qmb;
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c = 0.0;
    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    p_1[1]    = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1)
        target += watanabe_beta * log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t] + alpha*(lambda*r[t] + (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        s_int = ss[t];
        o_int = r[t] + 1;
        p_1[t+1] = p_o_1[s_int,o_int]*p_1[t] /
                   (p_o_1[s_int,o_int]*p_1[t] + p_o_0[s_int,o_int]*(1-p_1[t]));
        p_1[t+1] = (1-p_r)*p_1[t+1] + p_r*(1-p_1[t+1]);
        V1 = 0.8*p_1[t+1] + 0.2*(1-p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;
        Qmb[1,t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2,t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        Qnet[1,t+1] = Winf*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Winf*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
  }
}
generated quantities {
  real log_lik;
  {
    matrix[2,T] Qmb;
    matrix[2,T] Qmf;
    matrix[2,T] Qnet;
    matrix[2,T] V;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c = 0.0;
    real ll = 0.0;
    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;
    Qmb[1,1]  = 0; Qmb[2,1]  = 0;
    Qmf[1,1]  = 0; Qmf[2,1]  = 0;
    Qnet[1,1] = 0; Qnet[2,1] = 0;
    V[1,1]    = 0; V[2,1]    = 0;
    p_1[1]    = 0.5;
    for (t in 1:T) {
      if (tt[t] == 1)
        ll += log(1.0 / (1.0 + exp(-(Qnet[c[t],t] - Qnet[3-c[t],t]))));
      if (t < T) {
        Qmf[c[t],   t+1] = (1-alpha)*Qmf[c[t],t] + alpha*(lambda*r[t] + (1-lambda)*V[ss[t],t]);
        Qmf[3-c[t], t+1] = (1-forget)*Qmf[3-c[t],t];
        prev_c = (c[t] == 1) ? 0.5 : -0.5;
        s_int = ss[t];
        o_int = r[t] + 1;
        p_1[t+1] = p_o_1[s_int,o_int]*p_1[t] /
                   (p_o_1[s_int,o_int]*p_1[t] + p_o_0[s_int,o_int]*(1-p_1[t]));
        p_1[t+1] = (1-p_r)*p_1[t+1] + p_r*(1-p_1[t+1]);
        V1 = 0.8*p_1[t+1] + 0.2*(1-p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;
        Qmb[1,t+1] = PL[1]*V[1,t+1] + PL[2]*V[2,t+1];
        Qmb[2,t+1] = PR[1]*V[1,t+1] + PR[2]*V[2,t+1];
        Qnet[1,t+1] = Winf*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Winf*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
    log_lik = ll;
  }
}
"""


# ============================================================================
# Model registry
# ============================================================================

WBIC_MODELS = {
    'mf_p':      WBICModel(stan_code=_STAN_WBIC_MF_P,      param_names=['alpha', 'forget', 'lambda', 'Wmf', 'P'],                   r_is_int=False),
    'rac_p':     WBICModel(stan_code=_STAN_WBIC_RAC_P,     param_names=['alpha', 'tmp', 'P'],                                        r_is_int=True),
    'hyb_p':     WBICModel(stan_code=_STAN_WBIC_HYB_P,     param_names=['alpha', 'forget', 'lambda', 'Wmf', 'Wmb', 'P'],            r_is_int=False),
    'mb_p':      WBICModel(stan_code=_STAN_WBIC_MB_P,      param_names=['alpha', 'forget', 'Wmb', 'P'],                             r_is_int=False),
    'ls':        WBICModel(stan_code=_STAN_WBIC_LS,        param_names=['p_r', 'eps'],                                              r_is_int=False),
    'ls_asym_p': WBICModel(stan_code=_STAN_WBIC_LS_ASYM_P, param_names=['p_r', 'i_temp', 'P'],                                      r_is_int=True),
    'hyb_inf':   WBICModel(stan_code=_STAN_WBIC_HYB_INF,   param_names=['alpha', 'forget', 'lambda', 'Wmf', 'Winf', 'P', 'p_r'],   r_is_int=True),
}


# ============================================================================
# Compilation helper
# ============================================================================

def compile_wbic_models(model_keys):
    """Compile Stan models for the requested WBIC models.

    Returns a dict {model_key: compiled StanModel}.
    Compilation is slow (~30 s per model); call this once per session.
    """
    import pystan
    compiled = {}
    for key in model_keys:
        if key not in WBIC_MODELS:
            raise ValueError(f"Unknown WBIC model: '{key}'. Available: {list(WBIC_MODELS)}")
        print(f"Compiling Stan model: {key} ...", flush=True)
        compiled[key] = pystan.StanModel(model_code=WBIC_MODELS[key].stan_code)
        print(f"  done.", flush=True)
    return compiled


# ============================================================================
# Per-session fitting
# ============================================================================

def fit_wbic_session(sm, model_info, T, c, ss, tt, r, PR, PL,
                     n_iter=5000, n_warmup=750, n_chains=4, seed=123):
    """Fit a single session and return WBIC + mean posterior parameter estimates.

    Parameters
    ----------
    sm         : compiled pystan.StanModel (from compile_wbic_models)
    model_info : WBICModel namedtuple (from WBIC_MODELS)
    T, c, ss, tt, r, PR, PL : outputs of RL_data_arrange_single()
    n_iter, n_warmup, n_chains, seed : Stan sampling settings

    Returns
    -------
    dict with keys:
        'WBIC'      : float — Watanabe BIC for this session
        'n_trials'  : int   — number of trials (= T)
        <param_name>: float — posterior mean for each model parameter
    """
    r_data = r.astype(int) if model_info.r_is_int else r.astype(float)

    stan_data = {
        'T':  int(T),
        'PR': PR.astype(float),
        'PL': PL.astype(float),
        'c':  c.astype(int),
        'ss': ss.astype(int),
        'tt': tt.astype(int),
        'r':  r_data,
    }

    fit = sm.sampling(data=stan_data, iter=n_iter, warmup=n_warmup,
                      chains=n_chains, seed=seed, verbose=False)

    samples = fit.extract(permuted=True)
    log_lik = samples['log_lik']   # shape: (n_samples,)
    wbic    = float(-1.0 * np.mean(log_lik))

    result = {'WBIC': wbic, 'n_trials': int(T)}
    for p in model_info.param_names:
        result[p] = float(np.mean(samples[p]))

    return result
