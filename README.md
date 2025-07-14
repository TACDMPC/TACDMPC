# TACDMPC

[![CI](https://github.com/<user>/TACDMPC/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/TACDMPC/actions/workflows/ci.yml)

 **Transformer Actor–Critic with a differentiable MPC step.**

TACDMPC implements an Actor–Critic architecture where the policy delegates the first control action to a differentiable Model Predictive Controller.  Unlike the original paper that employs an MLP critic, here the critic is a small Transformer that processes a short history of states and actions.

## Overview

1. **Actor** – predicts a mean action and covariance; a Differentiable MPC module refines the first action.
2. **Critic** – Transformer encoder estimating the Q-value from past state/action pairs.
3. **Environment** – standard Gym/Gymnasium environments.

```mermaid
flowchart LR
    A[Actor Network] -- mean / std --> B(Diff-MPC) -- refined action --> C(Environment)
    C -- transition --> A
    C -- state,action history --> D[Critic Transformer]
```

## Installation

Create a Python ≥3.11 environment and install the package with optional extras for development and examples:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,examples]
```

The main runtime requirements are `torch`, `gymnasium`/`gym`, `numpy` and `tqdm`.  Optional extras provide `pytest`, `ruff`, `black` and `matplotlib` for plotting the demos.

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

## Testing

After installing the development extras simply run:

```bash
pytest -q
```

This checks gradient propagation and numerical consistency of the differentiable MPC.
