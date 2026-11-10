"""Central project configuration: filesystem paths and random keys.

Usage:
    from ritz_vs_pinn.config im port PATHS, KEYS, checkpoint_path, figure_path

    model_key = KEYS.model_init
    eqx.tree_serialise_leaves(checkpoint_path("pinn_poisson"), model)
    plt.savefig(figure_path("loss_curve"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from ritz_vs_pinn.loss_function import ritz_loss_fn, pinn_loss_fn

from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# config.py lives at <root>/src/ritz_vs_pinn/config.py
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT_DIR
    src: Path = ROOT_DIR / "src" / "ritz_vs_pinn"
    scripts: Path = ROOT_DIR / "scripts"
    checkpoints: Path = ROOT_DIR / "checkpoints"
    figures: Path = ROOT_DIR / "figures"
    results: Path = ROOT_DIR / "results"
    report: Path = ROOT_DIR / "report"
    report_figures: Path = ROOT_DIR / "report" / "figures"

    def make_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for p in (self.checkpoints, self.figures, self.results, self.report_figures):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def checkpoint_path(name: str, ext: str = ".eqx") -> Path:
    PATHS.checkpoints.mkdir(parents=True, exist_ok=True)
    return PATHS.checkpoints / f"{name}{ext}"


def run_dir(name: str, timestamp: bool = True) -> Path:
    """Create and return checkpoints/<name>[_<timestamp>]/."""
    folder = f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}" if timestamp else name
    path = PATHS.checkpoints / folder
    path.mkdir(parents=True, exist_ok=False)  # refuse to overwrite an existing run
    return path


def figure_path(name: str, ext: str = ".pdf") -> Path:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    return PATHS.figures / f"{name}{ext}"


def results_path(name: str, ext: str = ".npz") -> Path:
    PATHS.results.mkdir(parents=True, exist_ok=True)
    return PATHS.results / f"{name}{ext}"


# ---------------------------------------------------------------------------
# Random keys
# ---------------------------------------------------------------------------

SEED = 42


@dataclass(frozen=True)
class Keys:
    """Named PRNG keys derived from a single seed.

    Each key is derived with `fold_in(root, index)`, so adding a new named key
    never changes the values of existing ones (unlike sequential `split`s).
    """

    seed: int = SEED
    root: jax.Array = field(init=False)
    model_init: jax.Array = field(init=False)
    training: jax.Array = field(init=False)
    sampling: jax.Array = field(init=False)
    evaluation: jax.Array = field(init=False)

    # Stable indices — append new names, never reorder.
    _NAMES = ("model_init", "training", "sampling", "evaluation")

    def __post_init__(self) -> None:
        root = jr.PRNGKey(self.seed)
        object.__setattr__(self, "root", root)
        for i, name in enumerate(self._NAMES):
            object.__setattr__(self, name, jr.fold_in(root, i))

    def get(self, name: str) -> jax.Array:
        """Ad-hoc key derived from a string name (stable across runs)."""
        import zlib

        return jr.fold_in(self.root, zlib.crc32(name.encode()) & 0x7FFFFFFF)


KEYS = Keys()


def keys_for_seed(seed: int) -> Keys:
    """Fresh key set for a different seed, e.g. for repeated runs."""
    return Keys(seed=seed)


ACTIVATIONS = {
    "sigmoid": jax.nn.sigmoid,
    "tanh" : jax.nn.tanh,
    "relu" : jax.nn.relu
}

OPTIMIZERS = {
    "adam": optax.adam,
}

MMS_FUNCTIONS = {
    "1d_sinus_1" : {
        'f': lambda x: jnp.pi**2 / 4 * jnp.sin(jnp.pi * x / 2),
        'g': lambda x: jnp.sin(jnp.pi * x/2),
        'u': lambda x: jnp.sin(jnp.pi * x/2)
    },
    "2d_sinus_sums_1":{
        'f': lambda x: jnp.pi**2 / 4 * (jnp.sin(jnp.pi * x[0] / 2) + jnp.sin(jnp.pi * x[1] / 2)),
        'g': lambda x: jnp.sin(jnp.pi * x[0]/2) + jnp.sin(jnp.pi * x[1]/2),
        'u': lambda x: jnp.sin(jnp.pi * x[0]/2) + jnp.sin(jnp.pi * x[1]/2)
    },
    "general_sinus_sum": {
        'f': lambda x: jnp.pi**2 / 4 * jnp.sum(jnp.sin(jnp.pi * x / 2)),
        'g': lambda x: jnp.sum(jnp.sin(jnp.pi * x / 2)),
        'u': lambda x: jnp.sum(jnp.sin(jnp.pi * x / 2)),
    }
}

LOSS = {
    "ritz":ritz_loss_fn,
    "pinn":pinn_loss_fn
}