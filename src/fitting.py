"""
Shared fitting utilities for two-step task RL models.

Provides:
  - RL_data_arrange: convert pyControl sessions into Stan-compatible arrays
  - LOO: leave-one-out cross-validation (model-agnostic)
  - run_loo_all_mice: run LOO across multiple mice and return a DataFrame
"""
import numpy as np
import pandas as pd


def RL_data_arrange(sessions):
    """Convert a list of pyControl sessions into arrays for Stan fitting.

    Parses session.print_lines, where each line is space-separated with fields:
      [time, trial_num, ..., C:<0|1>, S:<0|1>, O:<0|1>, ..., CT:<FC|L|R>, ..., TS:<A|B>]

    Encoding:
      choice:      1 = left (C:1),  2 = right (C:0)
      second_state:1 = up (S:1),    2 = down (S:0)
      outcome:     1 = rewarded,    0 = unrewarded
      trial_type:  1 = free choice, 2 = forced choice

    Returns: S, T, c, ss, tt, r, PR, PL
    """
    S = len(sessions)
    T = []
    PL = np.zeros((2, S))
    PR = np.zeros((2, S))
    c  = np.zeros((S, 600))
    ss = np.zeros((S, 600))
    tt = np.zeros((S, 600))
    r  = np.zeros((S, 600)) - 1

    for i in range(S):
        b = sessions[i].print_lines
        n_trial = len(b)

        # Transition state is constant for the session — read from first trial
        trial = b[0].split(' ')
        trans_state = int(trial[8][3] == 'A')  # 1 if state A, 0 if state B
        if trans_state:
            PL[0, i] = 0.8;  PL[1, i] = 0.2
            PR[0, i] = 0.2;  PR[1, i] = 0.8
        else:
            PL[0, i] = 0.2;  PL[1, i] = 0.8
            PR[0, i] = 0.8;  PR[1, i] = 0.2

        choice       = np.ones(n_trial, dtype=int)
        second_state = np.ones(n_trial, dtype=int)
        outcome      = np.zeros(n_trial)
        trial_type   = np.ones(n_trial, dtype=int)

        for j in range(n_trial):
            d = b[j].split(' ')
            if d[4] == 'C:0':    choice[j] = 2        # right
            if d[5] == 'S:0':    second_state[j] = 2  # down
            outcome[j] = 1 if d[6] == 'O:1' else 0
            if d[9] != 'CT:FC':  trial_type[j] = 2    # forced choice

        T.append(n_trial)
        c[i,  :n_trial] = choice
        ss[i, :n_trial] = second_state
        r[i,  :n_trial] = outcome
        tt[i, :n_trial] = trial_type

    T   = np.array(T)
    T_max = max(T)
    c   = c[:,  :T_max]
    ss  = ss[:, :T_max]
    tt  = tt[:, :T_max]
    r   = r[:,  :T_max]

    return S, T, c, ss, tt, r, PR, PL


def RL_data_arrange_single(session):
    """Convert a single pyControl session into 1-D arrays for per-session Stan fitting.

    Same parsing logic as RL_data_arrange, but returns flat 1-D arrays
    (no session dimension) suitable for WBIC Stan models.

    Returns: T, c, ss, tt, r, PR, PL
        T  : int            — number of trials
        c  : int array [T]  — choices (1=left, 2=right)
        ss : int array [T]  — second states (1=up, 2=down)
        tt : int array [T]  — trial types (1=free choice, 2=forced)
        r  : float array [T] — outcomes (1.0=rewarded, 0.0=not)
        PR : float array [2] — right-choice transition probs [P(up), P(down)]
        PL : float array [2] — left-choice transition probs  [P(up), P(down)]
    """
    b = session.print_lines
    n_trial = len(b)

    # Transition state is constant for the session — read from first trial
    trial = b[0].split(' ')
    trans_state = int(trial[8][3] == 'A')   # 1 if state A, 0 if state B
    if trans_state:
        PL = np.array([0.8, 0.2])
        PR = np.array([0.2, 0.8])
    else:
        PL = np.array([0.2, 0.8])
        PR = np.array([0.8, 0.2])

    choice       = np.ones(n_trial, dtype=int)
    second_state = np.ones(n_trial, dtype=int)
    outcome      = np.zeros(n_trial)
    trial_type   = np.ones(n_trial, dtype=int)

    for j in range(n_trial):
        d = b[j].split(' ')
        if d[4] == 'C:0':    choice[j] = 2        # right
        if d[5] == 'S:0':    second_state[j] = 2  # down
        outcome[j] = 1.0 if d[6] == 'O:1' else 0.0
        if d[9] != 'CT:FC':  trial_type[j] = 2    # forced choice

    return n_trial, choice, second_state, trial_type, outcome, PR, PL


def LOO(model, sm, S, T, T_max, c, ss, tt, r, PL, PR):
    """Leave-one-out cross-validation.

    Args:
        model: a Model object from src.models (provides log_likelihood)
        sm: compiled pystan.StanModel.
            MAP vs MLE is determined by the Stan model itself:
            - MAP: compile with priors in the model block (default)
            - MLE: compile without priors in the model block

    Returns:
        nl: list of per-session normalized log-likelihoods
        param: parameters from the last fold
    """
    r_arr = r.astype(int) if model.r_is_int else r

    nl = []
    for i in range(S):
        # Test set: single held-out session
        test = {
            "T": T[i],
            "c": c[i, :].astype(int),
            "ss": ss[i, :].astype(int),
            "tt": tt[i, :].astype(int),
            "r": r_arr[i, :],
            "PL": PL[:, i],
            "PR": PR[:, i],
        }

        # Train set: all other sessions
        train = {
            "S": S - 1,
            "T_max": T_max,
            "T": np.delete(T, i),
            "PL": np.delete(PL, i, 1),
            "PR": np.delete(PR, i, 1),
            "c": np.delete(c.astype(int), i, 0),
            "ss": np.delete(ss.astype(int), i, 0),
            "tt": np.delete(tt.astype(int), i, 0),
            "r": np.delete(r_arr, i, 0),
        }

        param = sm.optimizing(data=train, seed=123, verbose=True)

        ll, n_fc = model.log_likelihood(
            param, test["c"], test["ss"], test["tt"], test["r"],
            test["PR"], test["PL"], test["T"]
        )
        nl.append(ll / n_fc)

    print(np.exp(np.mean(nl)))
    print("param", param)
    return nl, param


def run_loo_all_mice(model, sm, mice_data, estimate=0):
    """Run LOO for multiple mice and return a summary DataFrame.

    Args:
        model: a Model object
        sm: compiled pystan.StanModel
        mice_data: dict of {mouse_name: (S, T, T_max, c, ss, tt, r, PR, PL)}
        estimate: 0 = MAP, 1 = MLE

    Returns:
        df: DataFrame with columns [Mouse, Cross validation, params...]
    """
    rows = []
    all_results = {}

    for mouse_name, data in mice_data.items():
        S, T, T_max, c, ss, tt, r, PR, PL = data
        print(f"\n--- {mouse_name} (S={S}) ---")

        nl, param = LOO(model, sm, S, T, T_max, c, ss, tt, r, PL, PR, estimate)
        all_results[mouse_name] = (nl, param)

        row = {
            'Mouse': mouse_name,
            'Cross validation': np.exp(np.mean(nl)),
        }
        # Add all fitted parameters
        for p_name in model.param_names:
            if p_name in param:
                row[p_name] = param[p_name]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, all_results
