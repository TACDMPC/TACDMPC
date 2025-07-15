# TACDMPC


 **Transformer Actor–Critic with a differentiable MPC step.**

TACDMPC implements an Actor–Critic architecture where the policy operates in order to optimize the OCP weights and delegates the first control action to a differentiable Model Predictive Controller.  Unlike the original paper that employs an MLP critic, here the critic is a small Transformer that processes a short history of states and actions.

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

Runtime dependencies are `torch>=2.2`, `gymnasium`/`gym`, `numpy`, `tqdm` and `transformers`.  Optionally `cvxpylayers` can be installed if you prefer a CVXPY based solver.  Development extras provide `pytest`, `ruff` and `black` together with `matplotlib` for the demos.

## Running the examples

Each script inside `examples/` can be launched as a module:

```bash
python -m examples.01_demo_linear_tracking
python -m examples.02_demo_cartpole_regulation
python -m examples.03_demo_cartpole_regulationAC
python -m examples.01_simple
```

Alternatively a convenience entry point is installed:

```bash
ac-mpc-examples 03_demo_cartpole_regulationAC
```

## Testing

After installing the development extras simply run:

```bash
pytest -q
```

This checks gradient propagation and numerical consistency of the differentiable MPC.
