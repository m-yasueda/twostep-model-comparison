# Reinforcement Learning Strategies in a Two-Step Decision Task

Analysis code for studying the development of reinforcement learning strategies in wildtype mice performing a two-step decision task. Models are fit using Bayesian inference (PyStan), compared via leave-one-out cross-validation and WBIC, and validated through behavioral simulation.

## Overview

Six wildtype mice (WT1--WT6) were trained on a two-step Markov decision task across progressive training stages (4.2--4.8). We fit a family of RL models to choice behavior and compare them to characterize how decision strategies evolve with training.

### Models

| Model | Key | Parameters | Description |
|-------|-----|------------|-------------|
| Model-Free + Perseveration | `mf_p` | alpha, lambda, forget, Wmf, P | SARSA-like Q-learning with perseveration bias |
| Model-Based + Perseveration | `mb_p` | alpha, forget, Wmb, P | Transition-aware planning with perseveration |
| Hybrid + Perseveration | `hyb_p` | alpha, lambda, forget, Wmf, Wmb, P | Weighted MF + MB combination |
| Latent State | `ls_asym_p` | p_r, beta, P | Bayesian inference over hidden block states |
| Reward-as-Cue | `rac_p` | alpha, tmp, P | Reward outcome indexes next-trial Q-values |

All models with `_p` suffix include a choice perseveration parameter. Variants with `_pmulti` extend perseveration to separate parameters per session.

## Repository Structure

```
twostep-model-comparison/
├── config.py                   # Centralized paths and parameters
├── src/
│   ├── data_import.py          # pyControl session parser
│   ├── fitting.py              # LOO cross-validation engine
│   ├── wbic.py                 # WBIC fitting (Stan models + extraction)
│   └── models/                 # Model registry
│       ├── model_free.py       # mf, mf_p, mf_pmulti, p_only
│       ├── model_based.py      # mb, mb_p, mb_pmulti, mb_p_asym, ...
│       ├── hybrid.py           # hyb, hyb_p, hyb_pmulti
│       ├── latent_state.py     # ls_asym, ls_asym_p, ls_asym_pmulti
│       └── reward_as_cue.py    # rac_p
├── notebooks/                  # Analysis notebooks (see below)
├── data/raw/                   # Raw pyControl .txt session files
├── results/
│   ├── CV/                     # LOO cross-validation CSVs
│   ├── WBIC/                   # Per-session and per-subject WBIC CSVs
│   └── simulation/             # Behavioral simulation CSVs
└── figures/                    # Generated plots
```

## Notebooks

Run in order. Each notebook is self-contained once its prerequisites exist.

| # | Notebook | Description | Prerequisites |
|---|----------|-------------|---------------|
| 1 | `1_data_preprocessing.ipynb` | Parse raw pyControl `.txt` files into structured session data | Raw data in `data/raw/` |
| 1b | `1b_trial_counts.ipynb` | Summary of trial counts by stage and subject | Notebook 1 |
| 2 | `2_loo_cross_validation.ipynb` | Leave-one-out cross-validation for a single model and stage | Notebook 1 |
| 3 | `3_cv_model_comparison.ipynb` | Load CV results across models; comparison plots | Notebook 2 (all models) |
| 4 | `4_wbic_fitting.ipynb` | Fit WBIC per session using tempered posterior sampling | Notebook 1 |
| 5 | `5_wbic_model_comparison.ipynb` | Load WBIC results; model comparison plots | Notebook 4 (all models) |
| 5b | `5b_wbic_summary.ipynb` | Trial-normalized WBIC summary per subject | Notebook 4 |
| 6 | `6_behavioral_simulation.ipynb` | Simulate agents using MAP parameters; stay probability and trials-to-criterion analysis | Notebook 4 |
| 7 | `7_figures.ipynb` | Publication figures | Notebooks 3, 5, 6 |

### Typical workflow

```
Notebook 1  ──►  Notebook 2 (repeat per model)  ──►  Notebook 3
                 Notebook 4 (repeat per model)  ──►  Notebook 5 / 5b
                                                 ──►  Notebook 6
                                                           ──►  Notebook 7
```

## Training Stages

| Stage | Folder | Reward probabilities | Description |
|-------|--------|---------------------|-------------|
| 4.2--4.6 | `WT*_Training/` | 0.9 / 0.1 | Progressive training |
| 4.7 | `WT*_Training/` | 0.8 / 0.2 | Late training |
| 4.8 | `WT*/` (main) | 0.8 / 0.2 | Final experiment sessions |

Note: stage `4.8` is an analysis label. Sessions in the main folder have `training_stage == '4.7'` in the raw files but are treated as the final experiment phase.

## Quick Start

```python
import sys, os
sys.path.insert(0, '/path/to/twostep-model-comparison')
sys.path.insert(0, '/path/to/twostep-model-comparison/src')


from config import SUBJECTS, raw_subject_dir
from src.data_import import Experiment

# Load all sessions for one subject
exp = Experiment(raw_subject_dir('WT1'))
for session in exp.sessions:
    print(session.datetime_string, len(session.print_lines), 'trials')
```

## Dependencies

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib
- PyStan 2.x (Bayesian model fitting)

## Output Conventions

| File pattern | Contents |
|---|---|
| `results/CV/CV_{model}_{stage}.csv` | LOO cross-validation log-likelihood per session |
| `results/WBIC/WBIC_{model}_{stage}.csv` | Per-session WBIC + MAP parameters |
| `results/WBIC/WBIC_norm_subj_{model}_{stage}.csv` | Per-subject mean WBIC normalized by trial count |
| `results/simulation/sim_{model}_{stage}.csv` | Simulated stay probability and trials-to-criterion per session |
