"""
Model-Free (MF) family variants for the two-step task.

Variants:
  mf          - base model-free (alpha, forget, lambda, Wmf)
  mf_p        - + single perseveration (+ P)
  mf_pmulti   - + multi-trial perseveration via EMA (+ P, alpha_c)
  p_only      - perseveration only, no RL learning (P)
"""
import numpy as np
from . import Model, register

# ---------------------------------------------------------------------------
# Shared Stan data block (identical across all variants)
# ---------------------------------------------------------------------------
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

STAN_MF = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;

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

        Qnet[1,t+1] = Wmf*Qmf[1,t+1];
        Qnet[2,t+1] = Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

STAN_MF_P = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
   real P;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;

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

        Qnet[1,t+1] = Wmf*Qmf[1,t+1] + P*prev_c;
        Qnet[2,t+1] = Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

STAN_MF_PMULTI = _DATA_BLOCK + """
parameters {
   real<lower=0, upper=1> alpha;
   real<lower=0, upper=1> forget;
   real<lower=0, upper=1> lambda;
   real<lower=0> Wmf;
   real P;
   real<lower=0, upper=1> alpha_c;
}
model {
  alpha ~ beta(1.2,1.2);
  forget ~ beta(1.2,1.2);
  lambda ~ beta(1.2, 1.2);
  Wmf ~ gamma(2,0.4);
  P ~ student_t(4,0,2.5);
  alpha_c ~ beta(2, 2);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qmf;
    matrix[2,T[i]] Qnet;
    matrix[2,T[i]] V;
    real prev_c;
    real ema_c = 1.0;

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

        ema_c = (1 - alpha_c) * ema_c + alpha_c * prev_c;

        Qnet[1,t+1] = Wmf*Qmf[1,t+1] + P*ema_c;
        Qnet[2,t+1] = Wmf*Qmf[2,t+1];
      }
    }
  }
}
"""

STAN_P_ONLY = _DATA_BLOCK + """
parameters {
   real P;
}
model {
  P ~ student_t(4,0,2.5);

  for ( i in 1:S ) {
    matrix[2,T[i]] Qnet;
    real prev_c;

    Qnet[1,1] = 0.5;  Qnet[2,1] = 0.5;

    for ( t in 1:T[i] ) {
      if (tt[i, t] == 1){
        target += log(1.0/(1.0 + exp(-(Qnet[c[i,t],t] - Qnet[3-c[i,t],t]))));
      }
      if (t < T[i]){
        if (c[i,t] == 1){ prev_c = 0.5; }
        else if (c[i,t] == 2){ prev_c = -0.5; }

        Qnet[1,t+1] = P*prev_c;
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Python log-likelihood functions
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ll_mf(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
    V = np.zeros((2, n_trial))

    log_lik = 0.0
    for t in range(n_trial):
        Qnet_diff = Wmf * (Qmf[0, t] - Qmf[1, t])
        a_prob = _sigmoid(Qnet_diff)
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            Qmf[c[t]-1, t+1] = (1-alpha)*Qmf[c[t]-1, t] + alpha*(lamda*r[t] + (1-lamda)*V[ss[t]-1, t])
            Qmf[2-c[t], t+1] = (1-forget)*Qmf[2-c[t], t]
            V[ss[t]-1, t+1] = (1-alpha)*V[ss[t]-1, t] + alpha*r[t]
            V[2-ss[t], t+1] = (1-forget)*V[2-ss[t], t]
    return log_lik, n_freechoice


def ll_mf_p(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']
    P = param['P']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
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

            Qnet[0, t+1] = Wmf*Qmf[0, t+1] + P*prev_c
            Qnet[1, t+1] = Wmf*Qmf[1, t+1]
    return log_lik, n_freechoice


def ll_mf_pmulti(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    alpha = param['alpha']
    lamda = param['lambda']
    forget = param['forget']
    Wmf = param['Wmf']
    P = param['P']
    alpha_c = param['alpha_c']

    c = c[:n_trial]; ss = ss[:n_trial]; tt = tt[:n_trial]; r = r[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qmf = np.zeros((2, n_trial))
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

            prev_c = 0.5 if c[t] == 1 else -0.5
            ema_c = (1-alpha_c)*ema_c + alpha_c*prev_c

            Qnet[0, t+1] = Wmf*Qmf[0, t+1] + P*ema_c
            Qnet[1, t+1] = Wmf*Qmf[1, t+1]
    return log_lik, n_freechoice


def ll_p_only(param, c, ss, tt, r, PR, PL, n_trial):
    eps = 1e-16
    P = param['P']

    c = c[:n_trial]; tt = tt[:n_trial]
    n_freechoice = np.sum(tt == 1)
    Qnet = np.zeros((2, n_trial))
    Qnet[0, 0] = 0.5
    Qnet[1, 0] = 0.5

    log_lik = 0.0
    for t in range(n_trial):
        a_prob = _sigmoid(Qnet[0, t] - Qnet[1, t])
        if tt[t] == 1:
            log_lik += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1.0 - a_prob + eps)
        if t < n_trial - 1:
            prev_c = 0.5 if (c[t]-1) == 0 else -0.5
            Qnet[0, t+1] = P*prev_c
    return log_lik, n_freechoice


# ---------------------------------------------------------------------------
# Register all variants
# ---------------------------------------------------------------------------

register(Model(
    name="mf",
    family="model_free",
    description="Model-free baseline",
    stan_code=STAN_MF,
    log_likelihood=ll_mf,
    param_names=["alpha", "forget", "lambda", "Wmf"],
))

register(Model(
    name="mf_p",
    family="model_free",
    description="Model-free + single perseveration",
    stan_code=STAN_MF_P,
    log_likelihood=ll_mf_p,
    param_names=["alpha", "forget", "lambda", "Wmf", "P"],
))

register(Model(
    name="mf_pmulti",
    family="model_free",
    description="Model-free + multi-trial perseveration (EMA)",
    stan_code=STAN_MF_PMULTI,
    log_likelihood=ll_mf_pmulti,
    param_names=["alpha", "forget", "lambda", "Wmf", "P", "alpha_c"],
))

register(Model(
    name="p_only",
    family="model_free",
    description="Perseveration only (no RL learning)",
    stan_code=STAN_P_ONLY,
    log_likelihood=ll_p_only,
    param_names=["P"],
))
