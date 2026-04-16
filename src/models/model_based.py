"""
Model-Based (MB) family variants for the two-step task.

Variants:
  mb            - base model-based (alpha, forget, lambda, Wmb)
  mb_p          - + single perseveration (+ P)
  mb_pmulti     - + multi-trial perseveration via EMA (+ P, alpha_c)
  mb_p_asym     - asymmetric learning rates for reward/non-reward (alpha_r, alpha_n)
  mb_p_vinherit - value inheritance across sessions (alpha, forget, Wmb, P)
  mb_p_vasym    - value inheritance + asymmetric learning (alpha_r, alpha_n, forget, Wmb, P)
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

STAN_MB = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmb;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmb ~ gamma(2,0.4);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1];
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
  }
}
"""

STAN_MB_P = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmb;
   real P;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
  }
}
"""

STAN_MB_PMULTI = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmb;
   real P;
   real<lower=0, upper=1> alpha_c;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);
  alpha_c ~ beta(2, 2);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;
    real ema_c = 1.0;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * 0.5;
        } else if (c[i, t] == 2) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * (-0.5);
        }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + P*ema_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
  }
}
"""

STAN_MB_P_ASYM = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha_r;
   real<lower=0, upper=1> alpha_n;
   real<lower=0, upper=1> forget;
   real<lower=0> Wmb;
   real P;
}
model {
  alpha_r ~ beta(1.2,1.2);
  alpha_n ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

    Qmb[1,1] = 0;  Qmb[2,1] = 0;
    Qnet[1,1] = 0;  Qnet[2,1] = 0;
    V[1,1] = 0;  V[2,1] = 0;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        if (r[i, t] == 1){
          V[ss[i,t], t+1] = (1 - alpha_r) * V[ss[i,t], t] + alpha_r * r[i,t];
        } else {
          V[ss[i,t], t+1] = (1 - alpha_n) * V[ss[i,t], t] + alpha_n * r[i,t];
        }
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
  }
}
"""

STAN_MB_P_VINHERIT = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0> Wmb;
   real P;
}
model {
  vector[2] V_last;
  vector[2] Qnet_last;

  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

    if (i == 1){
      V[1,1] = 0.5;  V[2,1] = 0.5;
      Qnet[1,1] = 0.5;  Qnet[2,1] = 0.5;
      Qmb[1,1] = 0.5;  Qmb[2,1] = 0.5;
    } else {
      V[1,1] = V_last[1];  V[2,1] = V_last[2];
      Qnet[1,1] = Qnet_last[1];  Qnet[2,1] = Qnet_last[2];
      Qmb[1,1] = Qnet_last[1];  Qmb[2,1] = Qnet_last[2];
    }

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        V[ss[i,t], t+1] = (1 - alpha) * V[ss[i,t], t] + alpha * r[i,t];
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
    V_last[1] = V[1, T[i]];
    V_last[2] = V[2, T[i]];
    Qnet_last[1] = Qnet[1, T[i]];
    Qnet_last[2] = Qnet[2, T[i]];
  }
}
"""

STAN_MB_P_VASYM = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha_r;
   real<lower=0, upper=1> alpha_n;
   real<lower=0, upper=1> forget;
   real<lower=0> Wmb;
   real P;
}
model {
  vector[2] V_last;
  vector[2] Qnet_last;

  alpha_r ~ beta(1.2,1.2);
  alpha_n ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  Wmb ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmb;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

    if (i == 1){
      V[1,1] = 0.5;  V[2,1] = 0.5;
      Qnet[1,1] = 0.5;  Qnet[2,1] = 0.5;
      Qmb[1,1] = 0.5;  Qmb[2,1] = 0.5;
    } else {
      V[1,1] = V_last[1];  V[2,1] = V_last[2];
      Qnet[1,1] = Qnet_last[1];  Qnet[2,1] = Qnet_last[2];
      Qmb[1,1] = Qnet_last[1];  Qmb[2,1] = Qnet_last[2];
    }

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        if (r[i, t] == 1){
          V[ss[i,t], t+1] = (1 - alpha_r) * V[ss[i,t], t] + alpha_r * r[i,t];
        } else {
          V[ss[i,t], t+1] = (1 - alpha_n) * V[ss[i,t], t] + alpha_n * r[i,t];
        }
        V[3-ss[i,t],t+1] = (1 - forget) * V[3-ss[i,t], t];

        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qmb[1,t+1] = PL[1,i] * V[1,t+1] + PL[2,i] * V[2,t+1];
        Qmb[2,t+1] = PR[1,i] * V[1,t+1] + PR[2,i] * V[2,t+1];

        Qnet[1,t+1] = Wmb*Qmb[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmb*Qmb[2,t+1];
      }
    }
    V_last[1] = V[1, T[i]];
    V_last[2] = V[2, T[i]];
    Qnet_last[1] = Qnet[1, T[i]];
    Qnet_last[2] = Qnet[2, T[i]];
  }
}
"""


# ---------------------------------------------------------------------------
# Python log-likelihood functions
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ll_mb(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    forget = param['forget']
    Wmb = param['Wmb']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1]
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


def ll_mb_p(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    forget = param['forget']
    Wmb = param['Wmb']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
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
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            prev_c = 0.5 if (c[t]-1) == 0 else -0.5

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


def ll_mb_pmulti(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    forget = param['forget']
    Wmb = param['Wmb']
    P = param['P']
    alpha_c = param['alpha_c']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
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
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            if c[t] == 1:
                ema_c = (1-alpha_c)*ema_c + alpha_c*0.5
            elif c[t] == 2:
                ema_c = (1-alpha_c)*ema_c + alpha_c*(-0.5)

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + P*ema_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


def ll_mb_p_asym(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha_r = param['alpha_r']
    alpha_n = param['alpha_n']
    forget = param['forget']
    Wmb = param['Wmb']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
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
            if r[t] == 1:
                V[ss[t]-1, t+1] = (1-alpha_r)*V[ss[t]-1, t] + alpha_r*r[t]
            else:
                V[ss[t]-1, t+1] = (1-alpha_n)*V[ss[t]-1, t] + alpha_n*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            prev_c = 0.5 if (c[t]-1) == 0 else -0.5

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


def ll_mb_p_vinherit(param, c, ss, tt, r, PR, PL, n_trial):
    """Single-session log-likelihood for MB with value inheritance.
    Note: value inheritance is handled at the Stan level across sessions.
    For LOO cross-validation, this evaluates a single held-out session
    with initial values at 0.5 (matching first-session behavior).
    """
    eps = 1e-16
    alpha = param['alpha']
    forget = param['forget']
    Wmb = param['Wmb']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))
    # Initialize to 0.5 for value-inheritance variant
    Qnet[:, 0] = 0.5; V[:, 0] = 0.5; Qmb[:, 0] = 0.5
    prev_c = 0

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            prev_c = 0.5 if (c[t]-1) == 0 else -0.5

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


def ll_mb_p_vasym(param, c, ss, tt, r, PR, PL, n_trial):
    """Single-session log-likelihood for MB with value inheritance + asymmetric learning."""
    eps = 1e-16
    alpha_r = param['alpha_r']
    alpha_n = param['alpha_n']
    forget = param['forget']
    Wmb = param['Wmb']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmb = np.zeros((2, n_trial))
    Qnet = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))
    Qnet[:, 0] = 0.5; V[:, 0] = 0.5; Qmb[:, 0] = 0.5
    prev_c = 0

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            if r[t] == 1:
                V[ss[t]-1, t+1] = (1-alpha_r)*V[ss[t]-1, t] + alpha_r*r[t]
            else:
                V[ss[t]-1, t+1] = (1-alpha_n)*V[ss[t]-1, t] + alpha_n*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]

            prev_c = 0.5 if (c[t]-1) == 0 else -0.5

            Qmb[0, t+1] = PL[0]*V[0, t+1] + PL[1]*V[1, t+1]
            Qmb[1, t+1] = PR[0]*V[0, t+1] + PR[1]*V[1, t+1]

            Qnet[0, t+1] = Wmb*Qmb[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmb*Qmb[1, t+1]
    return log_lik, n_freechoice


# ---------------------------------------------------------------------------
# Register all variants
# ---------------------------------------------------------------------------

register(Model(
    name="mb",
    family="model_based",
    description="Model-based baseline",
    stan_code=STAN_MB,
    log_likelihood=ll_mb,
    param_names=["alpha", "forget", "lambda", "Wmb"],
))

register(Model(
    name="mb_p",
    family="model_based",
    description="Model-based + single perseveration",
    stan_code=STAN_MB_P,
    log_likelihood=ll_mb_p,
    param_names=["alpha", "forget", "lambda", "Wmb", "P"],
))

register(Model(
    name="mb_pmulti",
    family="model_based",
    description="Model-based + multi-trial perseveration (EMA)",
    stan_code=STAN_MB_PMULTI,
    log_likelihood=ll_mb_pmulti,
    param_names=["alpha", "forget", "lambda", "Wmb", "P", "alpha_c"],
))

register(Model(
    name="mb_p_asym",
    family="model_based",
    description="Model-based + perseveration, asymmetric learning rates",
    stan_code=STAN_MB_P_ASYM,
    log_likelihood=ll_mb_p_asym,
    param_names=["alpha_r", "alpha_n", "forget", "Wmb", "P"],
))

register(Model(
    name="mb_p_vinherit",
    family="model_based",
    description="Model-based + perseveration, values inherited across sessions",
    stan_code=STAN_MB_P_VINHERIT,
    log_likelihood=ll_mb_p_vinherit,
    param_names=["alpha", "forget", "Wmb", "P"],
))

register(Model(
    name="mb_p_vasym",
    family="model_based",
    description="Model-based + perseveration, value inheritance + asymmetric",
    stan_code=STAN_MB_P_VASYM,
    log_likelihood=ll_mb_p_vasym,
    param_names=["alpha_r", "alpha_n", "forget", "Wmb", "P"],
))
