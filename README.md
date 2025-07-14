# TACDMPC

Transformer Actor–Critic framework that integrates a differentiable Model
Predictive Controller (MPC) inside the policy network.  The actor predicts a
distribution over actions while a differentiable MPC module refines the first
control input.  The critic is built around a Transformer that estimates the
Q-value from a history of states and actions.

## Installation

Create a Python 3.11 environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

For development and testing you can also install the optional tools defined in
`pyproject.toml`:

```bash
pip install -e .[dev]
```

The main runtime dependencies are:

- `torch`
- `gymnasium` and `gym`
- `numpy`
- `tqdm`
- `matplotlib`
- `pytest` (for running the test suite)

## Examples

Several demos are available in the `examples/` directory.  The following command
starts a training session for the CartPole environment using the actor–critic
setup with differentiable MPC guidance:

```bash
python examples/03_demo_cartpole_regulationAC.py
```

## Running the tests

The repository includes a small test suite that checks gradient propagation and
correct forward behaviour.  After installing the development dependencies run:

```bash
pytest
```

