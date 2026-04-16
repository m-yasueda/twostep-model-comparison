"""
Centralized configuration for the RL Two-Step Task analysis pipeline.

All paths are relative to PROJECT_ROOT so notebooks work regardless of
where they are launched from. Import this module at the top of every
notebook/script:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import *
"""

import os

# ── Root paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DATA_ROOT = os.path.join(PROJECT_ROOT, '..', 'OIST')

# ── Data paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# ── Raw data: per-subject folders ─────────────────────────────────────────
SUBJECTS = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6']

def raw_subject_dir(subject, training=False):
    """Return path to a subject's raw data folder.

    Args:
        subject: e.g. 'WT1'
        training: if True, return the *_Training subfolder
    """
    suffix = '_Training' if training else ''
    return os.path.join(RAW_DATA_DIR, f'{subject}{suffix}')

# ── Processed CSV paths ──────────────────────────────────────────────────
STAGES = ['4.6', '4.7', '4.8']

def stage_sessions_csv(stage):
    """Return path to the sessions CSV for a given stage."""
    return os.path.join(PROCESSED_DATA_DIR, f'Stage{stage}_sessions.csv')

def stage_stay_csv(stage):
    """Return path to the stay CSV for a given stage."""
    return os.path.join(PROCESSED_DATA_DIR, f'Stage{stage}_stay.csv')

# ── Model paths ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
STAN_MODELS_DIR = os.path.join(MODELS_DIR, 'stan')
FITTED_MODELS_DIR = os.path.join(MODELS_DIR, 'fitted')

def stan_model_path(model_name):
    """Return path to a .stan file by name (without extension)."""
    return os.path.join(STAN_MODELS_DIR, f'{model_name}.stan')

def fitted_params_dir(model_name):
    """Return directory for fitted parameter .pkl files for a model."""
    d = os.path.join(FITTED_MODELS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d

# ── Cross-validation results ─────────────────────────────────────────────
CV_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'CV')
os.makedirs(CV_RESULTS_DIR, exist_ok=True)

def cv_result_csv(model_name, stage):
    """Return path to cross-validation result CSV."""
    return os.path.join(CV_RESULTS_DIR, f'CV_{model_name}_{stage}.csv')

def wbic_csv(model_name):
    """Return path to WBIC comparison CSV."""
    return os.path.join(CV_RESULTS_DIR, f'wbic_{model_name}.csv')

# ── Simulation paths ─────────────────────────────────────────────────────
SIMULATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'simulations')
os.makedirs(SIMULATION_DIR, exist_ok=True)

def simulation_csv(model_name, subject):
    """Return path to a simulation output CSV."""
    return os.path.join(SIMULATION_DIR, f'simulation_{model_name}_{subject}.csv')

def combined_simulation_csv(model_name):
    """Return path to aggregated simulation CSV with stay variable."""
    return os.path.join(SIMULATION_DIR, f'combined_data_with_stay_{model_name}.csv')

# ── WBIC results ─────────────────────────────────────────────────────────
WBIC_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'WBIC')
os.makedirs(WBIC_RESULTS_DIR, exist_ok=True)

def wbic_result_csv(model_name, stage):
    """Return path to per-session WBIC result CSV."""
    return os.path.join(WBIC_RESULTS_DIR, f'WBIC_{model_name}_{stage}.csv')

def wbic_subj_csv(model_name, stage):
    """Return path to per-subject summary WBIC CSV."""
    return os.path.join(WBIC_RESULTS_DIR, f'WBIC_subj_{model_name}_{stage}.csv')

def wbic_norm_subj_csv(model_name, stage):
    """Return path to per-subject summary CSV where WBIC is normalized by n_trials."""
    return os.path.join(WBIC_RESULTS_DIR, f'WBIC_norm_subj_{model_name}_{stage}.csv')

# ── Simulation results ────────────────────────────────────────────────────
SIM_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'simulation')
os.makedirs(SIM_RESULTS_DIR, exist_ok=True)

def sim_result_csv(model_name, stage):
    """Return path to simulation result CSV (per-session stay prob + TTC)."""
    return os.path.join(SIM_RESULTS_DIR, f'sim_{model_name}_{stage}.csv')

# ── Figures ──────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Source code ──────────────────────────────────────────────────────────
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# ── Transition probabilities (used by model fitting) ─────────────────────
# Common transition: 0.8 probability to matching state
# Rare transition: 0.2 probability to non-matching state
P_COMMON = 0.8
P_RARE = 0.2

# Default transition matrices (Right/Left choice -> state probabilities)
PR_DEFAULT = [P_RARE, P_COMMON]    # Right choice: [P(up), P(down)]
PL_DEFAULT = [P_COMMON, P_RARE]    # Left choice:  [P(up), P(down)]
