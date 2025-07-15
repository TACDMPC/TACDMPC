# TACDMPC


 **Transformer Actor–Critic with a differentiable MPC step.**


TACDMPC implements an Actor–Critic architecture where the actor embeds a differentiable MPC layer.  The MPC cost matrices ``Q`` and ``R`` are learned online and updated via back‑propagation.  The critic is a small Transformer built with the `transformers` library.


## Overview

1. **Actor** – operates on matrix weight of the DMPC in order to optimize them
2. **Critic** – Transformer encoder estimating the Q-value from past state/action pairs.


## Installation

Create a Python ≥3.11 environment and install the package with optional extras for development and examples:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,examples]
```

The main runtime requirements are `torch`, `gymnasium`/`gym`, `numpy`, `tqdm` and `transformers`.  Optional extras provide `pytest`, `ruff`, `black` and `matplotlib` for plotting the demos.

To run the plotting examples make sure the `examples` extra is installed:

```bash
pip install -e .[examples]
```


## Running the examples

Each script inside `examples/` can be launched as a module:

```bash
python -m examples.01_demo_linear_tracking
python -m examples.02_demo_cartpole_regulation
python -m examples.03_demo_cartpole_regulationAC
```

Alternatively a convenience entry point is installed:

```bash
ac-mpc-examples 03_demo_cartpole_regulationAC
```

## Reproducibility

A helper `utils.seed_everything(seed)` seeds Python's `random`, NumPy and
PyTorch for deterministic runs. All example scripts and the training loop call
this function with seed `0` by default. Pass a different value to vary the
random initialization.

## Testing

After installing the development extras simply run:

```bash
pytest -q
```

This checks gradient propagation and numerical consistency of the differentiable MPC.
