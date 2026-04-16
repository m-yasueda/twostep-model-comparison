"""
Latent-state family of RL models for the two-step task.

The agent tracks p_1: the Bayesian probability that the world is in state 1,
updated via Bayes' rule on each (second_state, reward) observation plus a
fixed block-reversal hazard rate p_r.  Action values are computed from the
state-value estimate and the known transition probabilities (softmax choice).

Variants  (matching notebook naming convention)
--------
ls_asym        : Latent State Assymetry          — p_r, i_temp
ls_asym_p      : Latent State Assymetry + P      — p_r, i_temp, P
ls_asym_pmulti : Latent State Assymetry + P multi — p_r, i_temp, P, alpha_c

Aliases (kept for backward compatibility)
---------
ls_p      → ls_asym_p
ls_pmulti → ls_asym_pmulti
"""
import numpy as np
from . import Model, register

# ---------------------------------------------------------------------------
# Shared reward-likelihood matrices (fixed task structure)
#   p_o_1[state, outcome]  state 0-indexed, outcome: 0=unrewarded, 1=rewarded
# ---------------------------------------------------------------------------
_REW_GOOD = 0.4
_REW_BAD  = 0.1
_NON_REW  = 0.5

_P_O_1 = np.array([[_NON_REW, _REW_BAD ],   # state=0 (up),   out=0/1
                    [_NON_REW, _REW_GOOD]])   # state=1 (down), out=0/1

_P_O_0 = np.array([[_NON_REW, _REW_GOOD],
                    [_NON_REW, _REW_BAD ]])


# ---------------------------------------------------------------------------
# Shared Bayesian state-update step
# ---------------------------------------------------------------------------
def _bayes_update(p1_t, s_int, o_int, p_r):
    p1 = (_P_O_1[s_int, o_int] * p1_t /
          (_P_O_1[s_int, o_int] * p1_t + _P_O_0[s_int, o_int] * (1.0 - p1_t)))
    return (1.0 - p_r) * p1 + p_r * (1.0 - p1)


# ---------------------------------------------------------------------------
# Python log-likelihood functions
# ---------------------------------------------------------------------------

def _ll_ls_asym(param, c, ss, tt, r, PR, PL, n_trial):
    """Latent State Assymetry — no perseveration (ls_asym)."""
    eps    = 1e-16
    p_r    = param['p_r']
    i_temp = param['i_temp']

    c  = c[:n_trial].astype(int)
    ss = ss[:n_trial].astype(int)
    tt = tt[:n_trial].astype(int)
    r  = r[:n_trial].astype(int)
    n_fc = int(np.sum(tt == 1))

    Q  = np.zeros((2, n_trial))
    p1 = np.zeros(n_trial)
    p1[0] = 0.5

    ll = 0.0
    for t in range(n_trial):
        a_prob = 1.0 / (1.0 + np.exp(-(Q[0, t] - Q[1, t])))
        if tt[t] == 1:
            ll += np.log(a_prob + eps) if (c[t] - 1) == 0 else np.log(1.0 - a_prob + eps)

        if t < n_trial - 1:
            s_int = int(ss[t]) - 1
            o_int = int(r[t])

            p1[t + 1] = _bayes_update(p1[t], s_int, o_int, p_r)

            V1 = _REW_GOOD * p1[t + 1] + _REW_BAD * (1.0 - p1[t + 1])
            V0 = 1.0 - V1

            Q[0, t + 1] = i_temp * (PL[0] * V0 + PL[1] * V1)
            Q[1, t + 1] = i_temp * (PR[0] * V0 + PR[1] * V1)

    return ll, n_fc


def _ll_ls_asym_p(param, c, ss, tt, r, PR, PL, n_trial):
    """Latent State Assymetry + P — single perseveration (ls_asym_p)."""
    eps    = 1e-16
    p_r    = param['p_r']
    i_temp = param['i_temp']
    P      = param['P']

    c  = c[:n_trial].astype(int)
    ss = ss[:n_trial].astype(int)
    tt = tt[:n_trial].astype(int)
    r  = r[:n_trial].astype(int)
    n_fc = int(np.sum(tt == 1))

    Q  = np.zeros((2, n_trial))
    p1 = np.zeros(n_trial)
    p1[0] = 0.5

    ll = 0.0
    for t in range(n_trial):
        a_prob = 1.0 / (1.0 + np.exp(-(Q[0, t] - Q[1, t])))
        if tt[t] == 1:
            ll += np.log(a_prob + eps) if (c[t] - 1) == 0 else np.log(1.0 - a_prob + eps)

        if t < n_trial - 1:
            s_int  = int(ss[t]) - 1
            o_int  = int(r[t])
            prev_c = 0.5 if c[t] == 1 else -0.5

            p1[t + 1] = _bayes_update(p1[t], s_int, o_int, p_r)

            V1 = _REW_GOOD * p1[t + 1] + _REW_BAD * (1.0 - p1[t + 1])
            V0 = 1.0 - V1

            Q[0, t + 1] = i_temp * (PL[0] * V0 + PL[1] * V1) + P * prev_c
            Q[1, t + 1] = i_temp * (PR[0] * V0 + PR[1] * V1)

    return ll, n_fc


def _ll_ls_asym_pmulti(param, c, ss, tt, r, PR, PL, n_trial):
    """Latent State Assymetry + P multi — EMA perseveration (ls_asym_pmulti)."""
    eps     = 1e-16
    p_r     = param['p_r']
    i_temp  = param['i_temp']
    P       = param['P']
    alpha_c = param['alpha_c']

    c  = c[:n_trial].astype(int)
    ss = ss[:n_trial].astype(int)
    tt = tt[:n_trial].astype(int)
    r  = r[:n_trial].astype(int)
    n_fc = int(np.sum(tt == 1))

    Q  = np.zeros((2, n_trial))
    p1 = np.zeros(n_trial)
    p1[0] = 0.5
    ema_c = 0.5

    ll = 0.0
    for t in range(n_trial):
        a_prob = 1.0 / (1.0 + np.exp(-(Q[0, t] - Q[1, t])))
        if tt[t] == 1:
            ll += np.log(a_prob + eps) if (c[t] - 1) == 0 else np.log(1.0 - a_prob + eps)

        if t < n_trial - 1:
            s_int = int(ss[t]) - 1
            o_int = int(r[t])

            ema_c = ((1.0 - alpha_c) * ema_c + alpha_c * 0.5  if c[t] == 1 else
                     (1.0 - alpha_c) * ema_c + alpha_c * (-0.5))

            p1[t + 1] = _bayes_update(p1[t], s_int, o_int, p_r)

            V1 = _REW_GOOD * p1[t + 1] + _REW_BAD * (1.0 - p1[t + 1])
            V0 = 1.0 - V1

            Q[0, t + 1] = i_temp * (PL[0] * V0 + PL[1] * V1) + P * ema_c
            Q[1, t + 1] = i_temp * (PR[0] * V0 + PR[1] * V1)

    return ll, n_fc


def _ll_ls(param, c, ss, tt, r,PR, PL, n_trial):
    eps = 1e-16
    
    p_r   = param['p_r']
    beta = param['beta']
    
    c = c[:n_trial]
    ss = ss[:n_trial]
    tt = tt[:n_trial]
    r  = r[:n_trial]
    n_freechoice = np.sum(tt == 1)    
    V       = np.zeros((2,n_trial))
    V[0,1] = 0.5
    V[1,1] = 0.5

    log_likelihood = 0.0
    for t in range (n_trial):        
        a_prob          = V[c[t]-1,t]*(1-beta) + beta*(1-V[c[t]-1, t])
        if (tt[t] == 1):
            log_likelihood += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1. - a_prob + eps)

        if t < n_trial-1 :
            if (ss[t]==1 and r[t] ==1): #up, rewarded
                V[1, t+1] = PL[0] * V[1, t] / (PL[0] *V[1, t] +  PL[1]  * (1 - V[1, t]))

            elif (ss[t] == 2 and r[t] == 1): #down, rewarded
                V[1, t+1] = PL[1]* V[1, t] / (PL[1]*V[1, t] +  PL[0] * (1 - V[1, t]))

            elif (ss[t]==1 and r[t] ==0):  #up, nonrewarded
                V[1, t+1] = PR[0]* V[1, t] / (PR[0]*V[1, t] +  PR[1] * (1 - V[1, t]))

            else:
                V[1, t+1] = PR[1]* V[1, t] / (PR[1]*V[1, t] +  PR[0] * (1 - V[1, t]))

            #Update of state probabilities due to possibility of block reversal.
            V[0, t+1] =(1 - p_r) * V[0, t+1] + p_r * (1 - V[0, t+1])
            V[1, t+1] = 1 - V[0,t+1]

    return log_likelihood, n_freechoice

def _ll_ls_P(param, c, ss, tt, r,PR, PL, n_trial):
    eps = 1e-16
    
    p_r   = param['p_r']
    beta = param['beta']
    P = param['P']
    
    c = c[:n_trial]
    ss = ss[:n_trial]
    tt = tt[:n_trial]
    r  = r[:n_trial]
    n_freechoice = np.sum(tt == 1)    
    V       = np.zeros((2,n_trial))
    V[0,1] = 0.5
    V[1,1] = 0.5
    prev_c = 0.5

    log_likelihood = 0.0
    for t in range (n_trial):        
        a_prob          = V[c[t]-1,t]*(1-beta) + beta*(1-V[c[t]-1, t])
        if (tt[t] == 1):
            log_likelihood += np.log(a_prob + eps) if (c[t]-1) == 0 else np.log(1. - a_prob + eps)

        if t < n_trial-1 :    
            if (c[t] == 1):
                prev_c = 0.5
            elif (c[t] == 2):
                prev_c = -0.5
                
            if (ss[t]==1 and r[t] ==1): #up, rewarded
                V[1, t+1] = PL[0] * V[1, t] / (PL[0] *V[1, t] +  PL[1]  * (1 - V[1, t]))

            elif (ss[t] == 2 and r[t] == 1): #down, rewarded
                V[1, t+1] = PL[1]* V[1, t] / (PL[1]*V[1, t] +  PL[0] * (1 - V[1, t]))

            elif (ss[t]==1 and r[t] ==0):  #up, nonrewarded
                V[1, t+1] = PR[0]* V[1, t] / (PR[0]*V[1, t] +  PR[1] * (1 - V[1, t]))

            else:
                V[1, t+1] = PR[1]* V[1, t] / (PR[1]*V[1, t] +  PR[0] * (1 - V[1, t]))

            #Update of state probabilities due to possibility of block reversal.
            V[0, t+1] =(1 - p_r) * V[0, t+1] + p_r * (1 - V[0, t+1]) +  P*prev_c
            V[1, t+1] = 1 - V[0,t+1]

    return log_likelihood, n_freechoice
# ---------------------------------------------------------------------------
# Stan code
# ---------------------------------------------------------------------------

_STAN_ASYM = """
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
  real<lower=0, upper=1> p_r;
  real<lower=0> i_temp;
}
model {
  p_r    ~ beta(1.2, 1.2);
  i_temp ~ gamma(2, 0.4);

  for (i in 1:S) {
    matrix[2, T[i]] V;
    matrix[2, T[i]] Qmb;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T[i]];
    int  s_int;
    int  o_int;
    real V1;

    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;

    V[1,1] = 0.5; V[2,1] = 0.5;
    Qmb[1,1] = 0.5; Qmb[2,1] = 0.5;
    p_1[1] = 0.5;

    for (t in 1:T[i]) {
      if (tt[i,t] == 1) {
        target += log(1.0 / (1.0 + exp(-(Qmb[c[i,t],t] - Qmb[3-c[i,t],t]))));
      }
      if (t < T[i]) {
        s_int = ss[i,t];
        o_int = r[i,t] + 1;

        p_1[t+1] = p_o_1[s_int,o_int] * p_1[t] /
                   (p_o_1[s_int,o_int] * p_1[t] + p_o_0[s_int,o_int] * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * p_1[t+1] + p_r * (1 - p_1[t+1]);

        V1 = 0.4 * p_1[t+1] + 0.1 * (1 - p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;

        Qmb[1,t+1] = i_temp * (PL[1,i]*V[1,t+1] + PL[2,i]*V[2,t+1]);
        Qmb[2,t+1] = i_temp * (PR[1,i]*V[1,t+1] + PR[2,i]*V[2,t+1]);
      }
    }
  }
}
"""

_STAN_ASYM_P = """
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
  real<lower=0, upper=1> p_r;
  real<lower=0> i_temp;
  real P;
}
model {
  p_r    ~ beta(1.2, 1.2);
  i_temp ~ gamma(2, 0.4);
  P      ~ student_t(4, 0, 2.5);

  for (i in 1:S) {
    matrix[2, T[i]] V;
    matrix[2, T[i]] Qmb;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T[i]];
    int  s_int;
    int  o_int;
    real V1;
    real prev_c;

    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;

    V[1,1] = 0.5; V[2,1] = 0.5;
    Qmb[1,1] = 0.5; Qmb[2,1] = 0.5;
    p_1[1] = 0.5;

    for (t in 1:T[i]) {
      if (tt[i,t] == 1) {
        target += log(1.0 / (1.0 + exp(-(Qmb[c[i,t],t] - Qmb[3-c[i,t],t]))));
      }
      if (t < T[i]) {
        s_int  = ss[i,t];
        o_int  = r[i,t] + 1;
        prev_c = (c[i,t] == 1) ? 0.5 : -0.5;

        p_1[t+1] = p_o_1[s_int,o_int] * p_1[t] /
                   (p_o_1[s_int,o_int] * p_1[t] + p_o_0[s_int,o_int] * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * p_1[t+1] + p_r * (1 - p_1[t+1]);

        V1 = 0.4 * p_1[t+1] + 0.1 * (1 - p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;

        Qmb[1,t+1] = i_temp * (PL[1,i]*V[1,t+1] + PL[2,i]*V[2,t+1]) + P * prev_c;
        Qmb[2,t+1] = i_temp * (PR[1,i]*V[1,t+1] + PR[2,i]*V[2,t+1]);
      }
    }
  }
}
"""

_STAN_ASYM_PMULTI = """
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
  real<lower=0, upper=1> p_r;
  real<lower=0> i_temp;
  real P;
  real<lower=0, upper=1> alpha_c;
}
model {
  p_r     ~ beta(1.2, 1.2);
  i_temp  ~ gamma(2, 0.4);
  P       ~ student_t(4, 0, 2.5);
  alpha_c ~ beta(2, 2);

  for (i in 1:S) {
    matrix[2, T[i]] V;
    matrix[2, T[i]] Qmb;
    matrix[2,2] p_o_1;
    matrix[2,2] p_o_0;
    real p_1[T[i]];
    int  s_int;
    int  o_int;
    real V1;
    real ema_c;

    p_o_1[1,1] = 0.5; p_o_1[1,2] = 0.1;
    p_o_1[2,1] = 0.5; p_o_1[2,2] = 0.4;
    p_o_0[1,1] = 0.5; p_o_0[1,2] = 0.4;
    p_o_0[2,1] = 0.5; p_o_0[2,2] = 0.1;

    V[1,1] = 0.5; V[2,1] = 0.5;
    Qmb[1,1] = 0.5; Qmb[2,1] = 0.5;
    p_1[1] = 0.5;

    ema_c = 0.5;
    for (t in 1:T[i]) {
      if (tt[i,t] == 1) {
        target += log(1.0 / (1.0 + exp(-(Qmb[c[i,t],t] - Qmb[3-c[i,t],t]))));
      }
      if (t < T[i]) {
        s_int = ss[i,t];
        o_int = r[i,t] + 1;

        if (c[i,t] == 1) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * 0.5;
        } else if (c[i,t] == 2) {
          ema_c = (1 - alpha_c) * ema_c + alpha_c * (-0.5);
        }

        p_1[t+1] = p_o_1[s_int,o_int] * p_1[t] /
                   (p_o_1[s_int,o_int] * p_1[t] + p_o_0[s_int,o_int] * (1 - p_1[t]));
        p_1[t+1] = (1 - p_r) * p_1[t+1] + p_r * (1 - p_1[t+1]);

        V1 = 0.4 * p_1[t+1] + 0.1 * (1 - p_1[t+1]);
        V[1,t+1] = 1 - V1;
        V[2,t+1] = V1;

        Qmb[1,t+1] = i_temp * (PL[1,i]*V[1,t+1] + PL[2,i]*V[2,t+1]) + P * ema_c;
        Qmb[2,t+1] = i_temp * (PR[1,i]*V[1,t+1] + PR[2,i]*V[2,t+1]);
      }
    }
  }
}
"""

_STAN_LS = """
data {
      int<lower=1> S;  // number of sessions
      int<lower=1> T_max;  // number of maximum number of trials in the dataframe
      int T[S]; // vector of number of trials at each session
      real PR[2,S];
      real PL[2,S];
      int<lower=0,upper=2> c[S, T_max]; // first stage choice
      int<lower=0,upper=2> ss[S, T_max]; // reached second state
      int<lower=0, upper=2> tt[S,T_max]; // whether it is forced- or free choice trial
      real r[S, T_max]; // reward
    }
    
parameters {
   real<lower =0, upper =1> p_r;    //the probability that agent believes that the state of the world changes on each step
   real<lower=0, upper = 0.5> beta;   // beta (eps)
}
model {
  //prior of free parameters

  p_r  ~ beta(2,2);    
  beta ~ gamma(2, 0.4);
  for ( i in 1:S ) {
    matrix[2,T[i]] V; // state values of second step states


    V[1,1] = 0.5; // state value for up state
    V[2,1] = 0.5; // state value for down state

    for ( t in 1:T[i] ) {
      // update likelihood function if trial type is free choice trial
      if (tt[i, t] == 1){
        target += log(V[c[i, t],t]*(1-beta) + beta*(1-V[c[i, t],t])); 
      }
      
      // update state value
      if (t < T[i]){
          if (ss[i, t]==1&& r[i, t] ==1){  //up, rewarded
              V[1, t+1] = PL[1,1] * V[1, t] / (PL[1,1] *V[1, t] +  PL[2,1]  * (1 - V[1, t]));
          }
          else if (ss[i, t] == 2 && r[i, t] == 1){//down, rewarded
              V[1, t+1] = PL[ 2,1]* V[1, t] / (PL[2,1]*V[1, t] +  PL[1,1] * (1 - V[1, t]));
          }
          else if (ss[i, t]==1 && r[i, t] ==0){ //up, nonrewarded
              V[1, t+1] = PR[1,1]* V[1, t] / (PR[ 1,1]*V[1, t] +  PR[2,1] * (1 - V[1, t]));
          }
          else{
              V[1, t+1] = PR[2,1]* V[1, t] / (PR[2,1]*V[1, t] +  PR[1,1] * (1 - V[1, t]));
          }
         
         //Update of state probabilities due to possibility of block reversal.
          V[1, t+1] =(1 - p_r) * V[1, t+1] + p_r * (1 - V[1, t+1]); 
          V[2, t+1] = 1 - V[1,t+1];        
      }
      
    }
  }
}
"""

_STAN_LS_P = """
data {
      int<lower=1> S;  // number of sessions
      int<lower=1> T_max;  // number of maximum number of trials in the dataframe
      int T[S]; // vector of number of trials at each session
      real PR[2,S];
      real PL[2,S];
      int<lower=0,upper=2> c[S, T_max]; // first stage choice
      int<lower=0,upper=2> ss[S, T_max]; // reached second state
      int<lower=0, upper=2> tt[S,T_max]; // whether it is forced- or free choice trial
      real r[S, T_max]; // reward
    }
    
parameters {
   real<lower =0, upper =1> p_r;    //the probability that agent believes that the state of the world changes on each step
   real<lower=0, upper = 1> beta;   // beta (eps)
    real<lower=0, upper = 1> P; // perservation parameter
}
model {
  //prior of free parameters
  //Mao
  //p_r  ~ beta(2,2);    
  //beta ~ gamma(2, 0.4);
  
  //Masa
  //p_r ~ beta(2,2);
  //beta ~ gamma(2,3);
  
  //Reddish
  p_r ~ beta(1.2,1.2);
  beta ~ exponential(0.5);
  P ~ student_t(4,0,2.5);
  

  
  for ( i in 1:S ) {
    matrix[2,T[i]] V; // state values of second step states
    real prev_c;

    prev_c = 0.5; // previous choice value, initialized to 0.5

    V[1,1] = 0.5; // state value for up state
    V[2,1] = 0.5; // state value for down state

    for ( t in 1:T[i] ) {
      // update likelihood function if trial type is free choice trial
      if (tt[i, t] == 1){
        target += log(V[c[i, t],t]*(1-beta) + beta*(1-V[c[i, t],t])); 
      }
      
      // update state value
      if (t < T[i]){
          if (ss[i, t]==1&& r[i, t] ==1){  //up, rewarded
              V[1, t+1] = PL[1,1] * V[1, t] / (PL[1,1] *V[1, t] +  PL[2,1]  * (1 - V[1, t]));
          }
          else if (ss[i, t] == 2 && r[i, t] == 1){//down, rewarded
              V[1, t+1] = PL[ 2,1]* V[1, t] / (PL[2,1]*V[1, t] +  PL[1,1] * (1 - V[1, t]));
          }
          else if (ss[i, t]==1 && r[i, t] ==0){ //up, nonrewarded
              V[1, t+1] = PR[1,1]* V[1, t] / (PR[ 1,1]*V[1, t] +  PR[2,1] * (1 - V[1, t]));
          }
          else{
              V[1, t+1] = PR[2,1]* V[1, t] / (PR[2,1]*V[1, t] +  PR[1,1] * (1 - V[1, t]));
          }
         
         //Update of state probabilities due to possibility of block reversal.
         if (c[i, t] == 1) {
            prev_c = -0.5;
          } else if (c[i, t] == 2) {
            prev_c = 0.5;
          }
          V[1, t+1] =(1 - p_r) * V[1, t+1] + p_r * (1 - V[1, t+1]); 
          V[2, t+1] = 1 - V[1,t+1] +  P*prev_c;     
      }
      
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register(Model(
    name           = 'ls_asym',
    family         = 'Latent State',
    description    = 'Latent State Assymetry (no perseveration)',
    stan_code      = _STAN_ASYM,
    log_likelihood = _ll_ls_asym,
    param_names    = ['p_r', 'i_temp'],
    r_is_int       = True,
))

register(Model(
    name           = 'ls_asym_p',
    family         = 'Latent State',
    description    = 'Latent State Assymetry + P (single perseveration)',
    stan_code      = _STAN_ASYM_P,
    log_likelihood = _ll_ls_asym_p,
    param_names    = ['p_r', 'i_temp', 'P'],
    r_is_int       = True,
))

register(Model(
    name           = 'ls_asym_pmulti',
    family         = 'Latent State',
    description    = 'Latent State Assymetry + P multi (EMA perseveration)',
    stan_code      = _STAN_ASYM_PMULTI,
    log_likelihood = _ll_ls_asym_pmulti,
    param_names    = ['p_r', 'i_temp', 'P', 'alpha_c'],
    r_is_int       = True,
))


register(Model(
    name           = 'ls',
    family         = 'Latent State',
    description    = 'Latent State (Akam 2015)',
    stan_code      = _STAN_LS,
    log_likelihood = _ll_ls,
    param_names    = ['p_r', 'beta'],
    r_is_int       = False,
))

register(Model(
    name           = 'ls_p',
    family         = 'Latent State',
    description    = 'Latent State (Akam 2015) + perseveration',
    stan_code      = _STAN_LS_P,
    log_likelihood = _ll_ls_P,
    param_names    = ['p_r', 'beta', 'P'],
    r_is_int       = False,
))

register(Model(
    name           = 'ls_pmulti',
    family         = 'Latent State',
    description    = 'alias for ls_asym_pmulti',
    stan_code      = _STAN_ASYM_PMULTI,
    log_likelihood = _ll_ls_asym_pmulti,
    param_names    = ['p_r', 'i_temp', 'P', 'alpha_c'],
    r_is_int       = True,
))
