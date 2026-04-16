"""
Model registry for two-step task RL models.

Usage:
    from src.models import get_model, list_models

    list_models()                    # see all available models
    model = get_model("mf_p")       # get model-free with perseveration
    model.stan_code                  # Stan model string
    model.log_likelihood(param, ...) # Python log-likelihood function
"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class Model:
    name: str
    family: str
    description: str
    stan_code: str
    log_likelihood: Callable
    param_names: List[str] = field(default_factory=list)
    r_is_int: bool = False  # True if Stan model declares r as int (LS, RAC families)


_REGISTRY = {}


def register(model):
    _REGISTRY[model.name] = model
    return model


def get_model(name):
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_models(family=None):
    print(f"{'Name':<25s} {'Family':<12s} Description")
    print("-" * 70)
    for name, m in sorted(_REGISTRY.items()):
        if family and m.family != family:
            continue
        print(f"  {name:<23s} {m.family:<12s} {m.description}")


# Import model modules to trigger registration
from . import model_free
from . import model_based
from . import hybrid
from . import latent_state
from . import reward_as_cue
