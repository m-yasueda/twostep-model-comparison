"""
Hybrid (MF+MB) family variants for the two-step task.

Variants:
  hyb         - base hybrid (alpha, forget, lambda, Wmf, Wmb)
  hyb_p       - + single perseveration (+ P)
  hyb_pmulti  - + multi-trial perseveration via EMA (+ P, alpha_c)
"""
import numpy as np
from . import Model, register

_DATA_BLOCK = """
data {
      int<lower=1> S;
      int<lower=1> T_max;
      int T[S];
      real PR[2,S];
      real PL[2,S];
      int<lower=0,upper=2> c[S, T_max];
      int<lower=0,upper=2> ss[S, T_max];
      int<lower=0, upper=2> tt[S,T_max];
      real r[S, T_max];
    }
"""

# ---------------------------------------------------------------------------
# Stan model code
# ---------------------------------------------------------------------------

STAN_HYB = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
   real<lower=0> Wmb;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);
  Wmb ~ gamma(2,0.4);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qmf[1,1] = 0;  Qmf[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        Qmf[c[i,t], t+1] = (1 - alpha) * Qmf[c[i,t], t] + alpha * (lambda * r[i,t] + (1 - lambda) * V[ss[i,t], t]);
        Qmf[3-c[i,t], t+1] = (1 - forget) * Qmf[3-c[i,t], t];
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + Wmf*Qmf[1,t+1];
        Qnet[2,t+1] = Wmb*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

STAN_HYB_P = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
   real<lower=0> Wmb;
   real P;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qmf[1,1] = 0;  Qmf[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        Qmf[c[i,t], t+1] = (1 - alpha) * Qmf[c[i,t], t] + alpha * (lambda * r[i,t] + (1 - lambda) * V[ss[i,t], t]);
        Qmf[3-c[i,t], t+1] = (1 - forget) * Qmf[3-c[i,t], t];
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

STAN_HYB_PMULTI = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
   real<lower=0> Wmb;
   real P;
   real<lower=0, upper=1> alpha_c;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);
  alpha_c ~ beta(2, 2);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real ema_c = 1.0;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qmf[1,1] = 0;  Qmf[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        Qmf[c[i,t], t+1] = (1 - alpha) * Qmf[c[i,t], t] + alpha * (lambda * r[i,t] + (1 - lambda) * V[ss[i,t], t]);
        Qmf[3-c[i,t], t+1] = (1 - forget) * Qmf[3-c[i,t], t];
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * 0.5;
        } else if (c[i, t] == 2) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * (-0.5);
        }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + Wmf*Qmf[1,t+1] + P*ema_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1] + Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

# STAN_HYB_INF uses int r (needed for Bayesian indexing) — separate data block
_DATA_BLOCK_INT_R = """
data {
  int<lower=1> S;
  int<lower=1> T_max;
  int T[S];
  real PR[2,S];
  real PL[2,S];
  int<lower=0,upper=2> c[S, T_max];
  int<lower=0,upper=2> ss[S, T_max];
  int<lower=0,upper=2> tt[S, T_max];
  int r[S, T_max];
}
"""

STAN_HYB_INF = """
data {
  int<lower=1> S;
  int<lower=1> T_max;
  int T[S];
  real PR[2,S];
  real PL[2,S];
  int<lower=0,upper=2> c[S, T_max];
  int<lower=0,upper=2> ss[S, T_max];
  int<lower=0,upper=2> tt[S, T_max];
  int r[S, T_max];
}
parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> forget;
  real<lower=0, upper=1> lambda;
  real<lower=0> Wmf;
  real<lower=0> Winf;
  real P;
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

  for (i in 1:S) {
    matrix[2,T[i]] V_mf;   // MF state values (TD-learned)
    matrix[2,T[i]] V_inf;  // asymmetric-inference state values (from p_1)
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qinf;
    matrix[2,T[i]] Qnet;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T[i]];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c;

    // Asymmetric-inference likelihood table (Blanco-Pozo 2024)
    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;

    V_mf[1,1]  = 0;   V_mf[2,1]  = 0;
    V_inf[1,1] = 0.5; V_inf[2,1] = 0.5;
    Qmf[1,1]   = 0;   Qmf[2,1]   = 0;
    Qinf[1,1]  = 0;   Qinf[2,1]  = 0;
    Qnet[1,1]  = 0;   Qnet[2,1]  = 0;
    p_1[1] = 0.5;

    for (t in 1:T[i]) {
      if (tt[i,t] == 1) {
        target += log(1.0 / (1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]) {
        // --- MF stream: TD update of V_mf and Qmf ---
        V_mf[ss[i,t],   t+1] = (1 - alpha)  * V_mf[ss[i,t],   t] + alpha * r[i,t];
        V_mf[3-ss[i,t], t+1] = (1 - forget) * V_mf[3-ss[i,t], t];

        Qmf[c[i,t],   t+1] = (1 - alpha)  * Qmf[c[i,t],   t]
                           + alpha * (lambda * r[i,t] + (1 - lambda) * V_mf[ss[i,t], t]);
        Qmf[3-c[i,t], t+1] = (1 - forget) * Qmf[3-c[i,t], t];

        // --- Asymmetric-inference stream: Bayesian p_1 update ---
        s_int = ss[i,t];
        o_int = r[i,t] + 1;

        p_1[t+1] = p_o_1[s_int, o_int] * p_1[t] /
                   (p_o_1[s_int, o_int] * p_1[t] + p_o_0[s_int, o_int] * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * p_1[t+1] + p_r * (1 - p_1[t+1]);

        V1 = 0.8 * p_1[t+1] + 0.2 * (1 - p_1[t+1]);
        V_inf[1, t+1] = 1 - V1;
        V_inf[2, t+1] = V1;

        Qinf[1, t+1] = PL[1,i] * V_inf[1, t+1] + PL[2,i] * V_inf[2, t+1];
        Qinf[2, t+1] = PR[1,i] * V_inf[1, t+1] + PR[2,i] * V_inf[2, t+1];

        // --- Perseveration & combine ---
        prev_c = (c[i,t] == 1) ? 0.5 : -0.5;
        Qnet[1, t+1] = Wmf * Qmf[1, t+1] + Winf * Qinf[1, t+1] + P * prev_c;
        Qnet[2, t+1] = Wmf * Qmf[2, t+1] + Winf * Qinf[2, t+1];
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Python log-likelihood functions
# This differs from the pure MF family which uses PLUS.
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ll_hyb(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']
    Wmb = param['Wmb']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            Qmf[c[t]-1, t+1] = (1-alpha)*Qmf[c[t]-1, t] + alpha*(lamda*r[t] + (1-lamda)*V[ss[t]-1, t])
            Qmf[2-c[t], t+1] = (1-forget)*Qmf[2-c[t], t]
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + Wmf*Qmf[0, t+1]
            Qnet[1, t+1] = Wmb*Qmb[1, t+1] + Wmf*Qmf[1, t+1]
    return log_lik, n_freechoice


def ll_hyb_p(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']
    Wmb = param['Wmb']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))
    prev_c = 0

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            Qmf[c[t]-1, t+1] = (1-alpha)*Qmf[c[t]-1, t] + alpha*(lamda*r[t] + (1-lamda)*V[ss[t]-1, t])
            Qmf[2-c[t], t+1] = (1-forget)*Qmf[2-c[t], t]
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            prev_c = 0.5 if (c[t]-1) == 0 else -0.5

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + Wmf*Qmf[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1] + Wmf*Qmf[1, t+1]
    return log_lik, n_freechoice


def ll_hyb_pmulti(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']
    Wmb = param['Wmb']
    P = param['P']
    alpha_c = param['alpha_c']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))
    ema_c = 1.0

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            Qmf[c[t]-1, t+1] = (1-alpha)*Qmf[c[t]-1, t] + alpha*(lamda*r[t] + (1-lamda)*V[ss[t]-1, t])
            Qmf[2-c[t], t+1] = (1-forget)*Qmf[2-c[t], t]
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            if c[t] == 1:
                ema_c = (1-alpha_c)*ema_c + alpha_c*0.5
            elif c[t] == 2:
                ema_c = (1-alpha_c)*ema_c + alpha_c*(-0.5)

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + Wmf*Qmf[0, t+1] + P*ema_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1] + Wmf*Qmf[1, t+1]
    return log_lik, n_freechoice

_REW_GOOD = 0.4
_REW_BAD  = 0.1
_NON_REW  = 0.5

_P_O_1 = np.array([[_NON_REW, _REW_BAD ],   # state=0 (up),   out=0/1
                    [_NON_REW, _REW_GOOD]])   # state=1 (down), out=0/1

_P_O_0 = np.array([[_NON_REW, _REW_GOOD],
                    [_NON_REW, _REW_BAD ]])

def _bayes_update(p1_t, s_int, o_int, p_r):
    p1 = (_P_O_1[s_int, o_int] * p1_t /
          (_P_O_1[s_int, o_int] * p1_t + _P_O_0[s_int, o_int] * (1.0 - p1_t)))
    return (1.0 - p_r) * p1 + p_r * (1.0 - p1)



def ll_hyb_inf(param, c, ss, tt, r, PR, PL, n_trial):
    """Hybrid (MF + Asymmetric Inference) + perseveration.

    MF stream: standard TD update of V_mf on second-step states, with Qmf
    eligibility trace through V_mf.
    Asymmetric-inference stream: Bayesian p_1 update using Blanco-Pozo 2024
    asymmetric likelihoods; V_inf derived from Pgood/Pbad scaling; Qinf
    computed via the first-step→second-step transition probabilities (PL/PR).
    Combined via Qnet = Wmf*Qmf + Winf*Qinf + P*prev_c (on left only).
    """
    eps = 1e-16
    p_r    = param['p_r']
    Winf   = param['Winf']
    alpha  = param['alpha']
    forget = param['forget']
    lamda  = param['lambda']
    Wmf    = param['Wmf']
    P      = param['P']

    c  = c[:n_trial]
    ss = ss[:n_trial]
    tt = tt[:n_trial]
    r  = r[:n_trial]
    n_freechoice = int(np.sum(tt == 1))

    V_mf  = np.zeros((2, n_trial))
    V_inf = np.zeros((2, n_trial))
    Qmf   = np.zeros((2, n_trial))
    Qinf  = np.zeros((2, n_trial))
    Qnet  = np.zeros((2, n_trial))

    V_inf[0, 0] = 0.5
    V_inf[1, 0] = 0.5

    p_1 = np.zeros(n_trial)
    p_1[0] = 0.5

    rew_good = 0.4
    rew_bad  = 0.1
    non_rew  = 0.5
    p_o_1 = np.array([[non_rew, rew_bad],
                      [non_rew, rew_good]])
    p_o_0 = np.array([[non_rew, rew_good],
                      [non_rew, rew_bad]])

    log_likelihood = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_likelihood += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)

        if t < n_trial - 1:
            # --- MF stream: TD update of V_mf and Qmf ---
            V_mf[ss[t]-1, t+1] = (1 - alpha)  * V_mf[ss[t]-1, t] + alpha * r[t]
            V_mf[2-ss[t], t+1] = (1 - forget) * V_mf[2-ss[t], t]

            Qmf[c[t]-1, t+1] = ((1 - alpha)  * Qmf[c[t]-1, t]
                                + alpha * (lamda * r[t] + (1 - lamda) * V_mf[ss[t]-1, t]))
            Qmf[2-c[t], t+1] = (1 - forget) * Qmf[2-c[t], t]

            # --- Asymmetric-inference stream: Bayesian p_1 update ---
            s_int = int(ss[t]) - 1
            o_int = int(r[t])
            p_1[t+1] = (p_o_1[s_int, o_int] * p_1[t] /
                        (p_o_1[s_int, o_int] * p_1[t] + p_o_0[s_int, o_int] * (1 - p_1[t])))
            p_1[t+1] = (1 - p_r) * p_1[t+1] + p_r * (1 - p_1[t+1])

            V1 = 0.8 * p_1[t+1] + 0.2 * (1 - p_1[t+1])
            V_inf[0, t+1] = 1 - V1
            V_inf[1, t+1] = V1

            Qinf[0, t+1] = PL[0] * V_inf[0, t+1] + PL[1] * V_inf[1, t+1]
            Qinf[1, t+1] = PR[0] * V_inf[0, t+1] + PR[1] * V_inf[1, t+1]

            # --- Perseveration & combine ---
            prev_c = 0.5 if (c[t]-1) == 0 else -0.5
            Qnet[0, t+1] = Wmf * Qmf[0, t+1] + Winf * Qinf[0, t+1] + P * prev_c
            Qnet[1, t+1] = Wmf * Qmf[1, t+1] + Winf * Qinf[1, t+1]

    return log_likelihood, n_freechoice



# ---------------------------------------------------------------------------
# Register all variants
# ---------------------------------------------------------------------------

register(Model(
    name="hyb",
    family="hybrid",
    description="Hybrid (MF+MB) baseline",
    stan_code=STAN_HYB,
    log_likelihood=ll_hyb,
    param_names=["alpha", "forget", "lambda", "Wmf", "Wmb"],
))

register(Model(
    name="hyb_p",
    family="hybrid",
    description="Hybrid + single perseveration",
    stan_code=STAN_HYB_P,
    log_likelihood=ll_hyb_p,
    param_names=["alpha", "forget", "lambda", "Wmf", "Wmb", "P"],
))

register(Model(
    name="hyb_pmulti",
    family="hybrid",
    description="Hybrid + multi-trial perseveration (EMA)",
    stan_code=STAN_HYB_PMULTI,
    log_likelihood=ll_hyb_pmulti,
    param_names=["alpha", "forget", "lambda", "Wmf", "Wmb", "P", "alpha_c"],
))

register(Model(
    name="hyb_inf",
    family="hybrid",
    description="Asymmetric Inference + Model-Free + single perseveration",
    stan_code=STAN_HYB_INF,
    log_likelihood=ll_hyb_inf,
    param_names=["alpha", "forget", "lambda", "Wmf", "Winf", "P", "p_r"],
    r_is_int=True,
))
