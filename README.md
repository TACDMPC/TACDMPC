# TACDMPC

**Transformer Actor–Critic with a differentiable MPC step.**

TACDMPC implements an Actor–Critic architecture where the policy learns the MPC cost weights online and delegates the first control action to a differentiable Model Predictive Controller. The critic is a lightweight Transformer built with the Hugging Face `transformers` library.

## Overview

1. **Actor** – learns the MPC cost matrices $Q$ and $R$ (also time varying) and updates them via backpropagation through the solver.
2. **Critic** – Transformer encoder estimating the state–action value from past and predicted trajectories.

## Installation

Create a Python ≥3.11 environment and install the package with optional extras for development and examples:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,examples]
```

Runtime requirements include **PyTorch ≥2.2** and **transformers**. Optional extras provide `pytest`, `ruff`, `black` and `matplotlib` for the demos.

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
