# Transformer Actor-Critic with Differentiable MPC (TACDMPC)

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**TACDMPC** is a PyTorch-based library for implementing Actor-Critic style reinforcement learning, where the actor's policy is represented by a **differentiable Model Predictive Control (MPC)** layer.

This approach, inspired by the paper ["Actor-Critic Model Predictive Control"](https://arxiv.org/abs/2306.09852), bridges the gap between model-free reinforcement learning and classical model-based control. Instead of learning a direct mapping from observation to action, the actor learns to parameterize the cost function of an MPC, which then solves for the optimal action at each step. This endows the agent with the predictive power and robustness of MPC while retaining the flexibility and end-to-end learning capabilities of RL.

---

## Core Concept: How It Works

The key idea is to replace the standard policy network (e.g., an MLP that outputs an action) with a more structured, physics-aware policy.

1.  **The Actor is a Cost Predictor**: The actor's neural network (named `cost_map_net` in the code) does not output an action directly. Instead, it takes an observation from the environment and outputs the **cost function parameters** (matrices `C` and vectors `c`) for a quadratic MPC cost function.
2.  **The DMPC is the Action Generator**: A differentiable MPC layer (`DifferentiableMPCController`) takes the predicted cost matrices and the current physical state of the system. Using a known dynamics model (`f_dyn`), it solves a short-horizon optimal control problem, producing the optimal action sequence. The first action of this sequence is executed in the environment.
3.  **End-to-End Training**: Because the entire MPC solve is implemented as a differentiable PyTorch layer (using `torch.autograd.Function`), gradients from the RL algorithm (PPO) can flow back through the controller and into the actor's neural network. This allows the actor to learn *how to specify goals* (i.e., the costs) for the MPC to achieve the overall task objective.
4.  **The Critic provides the Learning Signal**: A Transformer-based Critic evaluates the states visited by the agent, providing the advantage estimates needed to guide the actor's learning process.

---

### Architectural Deep Dive: The Transformer Critic

A key innovation in this library is the use of a Transformer architecture for the Critic network. This choice is motivated by a modern perspective on reinforcement learning that reframes it as a sequence modeling problem.

**Beyond the Markov Assumption**

Traditional RL often relies on the Markov assumption: the idea that the current state \\$s_t\\$ contains all the information needed to make an optimal decision. However, many real-world problems are **Partially Observable Markov Decision Processes (POMDPs)**, where history is crucial.

* MLP/CNN-based policies are *memoryless*; they only see the current state \\$s_t\\$.
* RNN-based policies attempt to solve this by compressing the entire past into a fixed-size hidden state \\$h_t = f(h_{t-1}, s_t)\\$, which can lead to information loss.

**Transformers as "Memory-Rich" Function Approximators**

A Transformer does not need to compress history. By using its self-attention mechanism, it can directly access and weigh the importance of specific, salient past events from a long context window. This makes it an exceptionally powerful function approximator for RL policies and value functions.

In this library, we leverage this capability for the Critic network:

* **Input as a Sequence**: Instead of just the current state \\$s_t\\$, the `CriticTransformer` receives a **sequence of recent states**: (..., \\$s_{t-2}, s_{t-1}, s_t\\$).
* **Contextual Value Estimation**: The Critic uses self-attention to analyze this recent trajectory. It can learn to identify salient past events (e.g., states leading to instability or high rewards) to produce a more accurate and context-aware value estimate \\$V(\text{history}_t)\\$.
* **Adaptive Behavior through Memory**: This architecture unlocks qualitatively new capabilities. An agent equipped with such a policy can adapt its strategy **within a single episode**. For instance, after stumbling on a new type of terrain, the agent can "remember" this event (as it is still in its context window) and adjust its behavior when it encounters that terrain again moments later. This is a fundamental step towards truly adaptive systems.

### Architectural Flowchart
TODO
## Key Features

* **`ActorMPC`**: A module whose neural network learns a map from environment observations to MPC cost function parameters. This allows the policy to adapt its short-term objectives based on the current context. It elegantly handles the difference between the observation dimension (used by the network) and the physical state dimension (used by the MPC's dynamics model).
* **`CriticTransformer`**: A value function estimator based on `transformers.BertModel`. Its attention mechanism is well-suited to processing sequences of states to accurately estimate expected returns.
* **`DifferentiableMPCController`**: A fully-batched MPC solver implemented as a `torch.autograd.Function` for end-to-end differentiation. It uses a user-provided system dynamics model to find optimal control actions and supports linearization via analytic Jacobians (`analytic`), auto-differentiation (`auto_diff`), or finite differences (`finite_diff`).
* **Training Framework**: The training is handled by a PPO (Proximal Policy Optimization) loop in `training_loop.py`. It includes:
    * Parallel environment management with `ParallelEnvManager`.
    * Advantage calculation using GAE (Generalized Advantage Estimation).
    * Use of the **MPVE (Model Predictive Value Expansion) loss**, a technique that leverages the MPC's short-term predictions to provide an additional, more stable learning signal to the critic.
    * Checkpointing with `CheckpointManager`.
* **Koopman Autoencoders (Optional)**: Includes tools for system identification using Koopman theory. This allows for approximating a **globally linear** system model from trajectory data. The learned model can then be supplied to the `DifferentiableMPCController`, enabling the application of this framework even to complex systems for which no analytical model is available.

---

## Installation

A Python version of **3.11+** is recommended.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
    ```

2.  **Install the package in editable mode:**
    The `[dev,examples]` extra includes all dependencies for development, testing, and running the included examples.
    ```bash
    pip install -e .[dev,examples]
    ```
    The main runtime requirements are `torch`, `gym`, `numpy`, and `transformers`.

---

## Running the Examples 

The example scripts in the `examples/` directory demonstrate the library's capabilities on various control problems. To run an example, execute it as a Python module from the root of the repository:

```bash
# Example 1: Regulate a double integrator system
python -m examples.01_demo_linear_AC

# Example 2: Stabilize a CartPole system
python -m examples.02_demo_cartpole_AC
```
![Benchmark](https://raw.githubusercontent.com/TACDMPC/TACDMPC/MASTER/examples/evaluation_results_cartpole_acmpc.png)
```bash
# Example 3: Train a unicycle to follow a trajectory NOT FULLY IMPLEMENTED 
python -m examples.03_demo_unicycle_AC
```

---

## Testing
NOTE THE TEST REFER TO AN OLD VERSION NOT WORKING ANYMORE WILL BE UPDATED AS CORE LOGIC IS FINALIZED
```bash
pytest
```

---

## Citation

If you find this work useful for your research, please consider citing the papers that inspired this framework:

```bibtex
@article{romero2023actorcritic,
  title={Actor-Critic Model Predictive Control},
  author={Romero, Angel and Aljalbout, Elie and Song, Yunlong and Scaramuzza, Davide},
  journal={arXiv preprint arXiv:2306.09852},
  year={2023}
}
@misc{amos2019differentiablempcendtoendplanning,
      title={Differentiable MPC for End-to-end Planning and Control}, 
      author={Brandon Amos and Ivan Dario Jimenez Rodriguez and Jacob Sacks and Byron Boots and J. Zico Kolter},
      year={2019},
      eprint={1810.13400},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/1810.13400](https://arxiv.org/abs/1810.13400)}, 
}
```
