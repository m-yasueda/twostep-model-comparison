"""
Reward-as-cue family of RL models for the two-step task.

The agent learns Q_td[action, prev_state, prev_outcome] — action values
conditioned on the previous second-step state and reward outcome used as a
contextual cue for the current trial.

Variants
--------
rac_p : alpha, tmp, P  (single perseveration; no pmulti variant)
"""
import numpy as np
from . import Model, register


# ---------------------------------------------------------------------------
# Python log-likelihood
# ---------------------------------------------------------------------------

def _ll_rac_p(param, c, ss, tt, r, PR, PL, n_trial):
    """Reward-as-cue + single perseveration (rac_p)."""
    eps   = 1e-16
    alpha = param['alpha']
    beta  = param['tmp']
    P     = param['P']

    c  = c[:n_trial].astype(int)
    ss = ss[:n_trial].astype(int)
    tt = tt[:n_trial].astype(int)
    r  = r[:n_trial].astype(int)
    n_fc = int(np.sum(tt == 1))

    Q_td  = np.zeros((2, 2, 2))   # [action, prev_state, prev_outcome]
    q1    = [0.5, 0.5]
    prev_c = 0.0
    ps, po = 0, 0                 # initial cue state (0-indexed)

    ll = 0.0
    for t in range(n_trial):
        a_prob = 1.0 / (1.0 + np.exp(-beta * (q1[0] - q1[1])))
        if tt[t] == 1:
            ll += np.log(a_prob + eps) if (c[t] - 1) == 0 else np.log(1.0 - a_prob + eps)

        if t < n_trial - 1:
            prev_c = 0.5 if (c[t] - 1) == 0 else -0.5

            Q_td[c[t] - 1, ps, po] = ((1.0 - alpha) * Q_td[c[t] - 1, ps, po]
                                       + alpha * r[t])
            ps = ss[t] - 1        # 0-indexed second state
            po = int(r[t])        # 0 or 1

            q1[0] = Q_td[0, ps, po] + P * prev_c
            q1[1] = Q_td[1, ps, po]

    return ll, n_fc


# ---------------------------------------------------------------------------
# Stan code  (init_ps / init_po removed; hardcoded to ps=1, po=1)
# ---------------------------------------------------------------------------

_STAN_RAC_P = """
data {
  int<lower=1> S;
  int<lower=1> T_max;
  int T[S];
  real PR[2, S];
  real PL[2, S];
  int<lower=0,upper=2> c[S, T_max];
  int<lower=0,upper=2> ss[S, T_max];
  int<lower=0,upper=2> tt[S, T_max];
  int r[S, T_max];
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

  for (i in 1:S) {
    vector[2] Q;
    real Q_td[2, 2, 2];
    real prev_c;
    int  ps;
    int  po;
    int  c_int;

    Q[1] = 0.5; Q[2] = 0.5;
    for (a in 1:2)
      for (s in 1:2)
        for (o in 1:2)
          Q_td[a,s,o] = 0.5;

    ps = 1; po = 1;   // fixed initial cue state

    for (t in 1:T[i]) {
      if (tt[i,t] == 1) {
        target += log(1.0 / (1e-16 + 1.0 + exp(-tmp * (Q[c[i,t]] - Q[3-c[i,t]]))));
      }
      if (t < T[i]) {
        prev_c = (c[i,t] == 1) ? 0.5 : -0.5;
        c_int  = c[i,t];
        Q_td[c_int, ps, po] = (1.0 - alpha) * Q_td[c_int, ps, po] + alpha * r[i,t];
        ps = ss[i,t];
        po = r[i,t] + 1;
        Q[1] = Q_td[1, ps, po] + P * prev_c;
        Q[2] = Q_td[2, ps, po];
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(Model(
    name        = 'rac_p',
    family      = 'Reward as Cue',
    description = 'Reward-as-cue TD learning + single perseveration',
    stan_code   = _STAN_RAC_P,
    log_likelihood = _ll_rac_p,
    param_names = ['alpha', 'tmp', 'P'],
    r_is_int    = True,
))
