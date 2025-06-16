<p align="center">
  <img src="assets\rl-video-episode-1200.gif" alt="Project Banner" width="200" />
</p>
<h3 align="center">SAC implementation</h3>

## Table of Contents
- [Introduction](#introduction)
- [Structure](#structure)
- [Quick Start](#quick-start)
- [Method](#method)
  - [Differences between Q1, Q2, and Q3](#differences-between-q1-q2-and-q3)
  - [High Light](#high-light)
- [Result](#result)

# 📌 Introduction

This repository contains the implementation of reinforcement learning agents for solving continuous control tasks, focusing on applying reinforcement learning algorithms to environments with continuous action spaces, increasing in complexity from a simple inverted pendulum to a complex humanoid robot.

| Environment        | Description                                       | Demonstration                                        |
|--------------------|---------------------------------------------------|------------------------------------------------------|
| **Pendulum**       | Control a simple inverted pendulum system.        | <img src="assets\Pendulum-example.gif" alt="Pendulum" width="172" />                     |
| **CartPole Balance** | Balance a pole on a moving cart.                  | <img src="assets\CartPole-example.gif" alt="CartPole Balance" width="172" />     |
| **Humanoid Walk**  | Control a complex humanoid robot to walk.         | <img src="assets\rl-video-episode-1200.gif" alt="Humanoid Walk" width="172" />          |

# 📁 Structure

Here is a representation of the project directory structure:

```bash
📦DRL-Assignment-4
 ┣ 📂Q1 # Pendulum
 ┃ ┣ 📜sac.ipynb
 ┃ ┣ 📜student_agent.py
 ┃ ┣ 📜eval.py
 ┃ ┗ 📂[model_file]
 ┣ 📂Q2 # CartPole Balance
 ┃ ┣ 📜sac.ipynb
 ┃ ┣ 📜student_agent.pyimplementation
 ┃ ┣ 📜eval.py
 ┃ ┗ 📂[model_file]
 ┣ 📂Q3 # Humanoid Walk
 ┃ ┣ 📜train.py
 ┃ ┣ 📜student_agent.py
 ┃ ┣ 📜eval.py
 ┃ ┗ 📂[model_file]
 ┣ 📜dmc.py # DeepMind Control Suite
 ┗ 📜requirements.txt # Project dependencies
```

- Training code: `sac.ipynb` and `train.py`.
- Evaluation code: `student_agent.py`, `eval.py`, and `[model_file]`.

#  🚀 Quick Start

To get started with this project, follow these steps:

1.  **Create and activate the Conda environment:**
    ```bash
    conda create -n rl-sac python==3.10 -y
    conda activate rl-sac
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Test your code locally:**
    ```bash
    python eval.py
    ```

4.  **GitHub Actions Workflow:**
    This repository is set up with a GitHub Actions workflow that automatically evaluates your `student_agent.py` when you push new commits. Ensure the workflow is enabled in your forked repository's "Actions" tab.

# 💡 Method

## 🛠️ Differences between Q1, Q2, and Q3

The training scripts evolve across the three tasks to handle the increasing complexity and leverage more advanced features:

-   **Q1 (Pendulum):** The training is implemented in a Jupyter Notebook (`sac.ipynb`) and uses the standard `gym.make` for environment creation. It focuses on the basic SAC implementation.
-   **Q2 (CartPole Balance):** The training is also in a Jupyter Notebook (`sac.ipynb`) but switches to using `dmc.make_dmc_env` for the DeepMind Control Suite environment. It introduces the use of `gym.vector.SyncVectorEnv` for vectorized environments to speed up data collection.
-   **Q3 (Humanoid Walk):** The training is implemented in a Python script (`train.py`). It continues to use `dmc.make_dmc_env` and `gym.vector.SyncVectorEnv`. This script includes more sophisticated features like orthogonal weight initialization, more detailed checkpointing logic (saving the best agent and on interrupt), and video recording during training.

## ✨ Highlights

-   **Warm-up:** The training process includes a warm-up phase where the agent interacts with the environment using a random policy for a specified number of steps (`warm_up_steps`). This helps populate the replay buffer with initial diverse experiences before the agent starts learning from the data.
-   **TD3 style update:** The SAC agent implementation incorporates elements inspired by the TD3 (Twin Delayed DDPG) algorithm, which is well-suited for continuous action spaces. Key features include using two Q-networks to mitigate overestimation bias and delaying policy updates relative to Q-function updates. The frequency of these updates is controlled by the `policy_update_freq` and `target_update_freq` parameters in the `SACAgent`.
-   **JIT:** The implementation in Q1 and Q2 demonstrates the use of Just-In-Time (JIT) compilation with `torch.jit.script` and `torch.jit.trace`. JIT compilation can improve performance by compiling Python code into a more efficient representation. Note that `use_jit` is set to `False` by default in `Q3/train.py`.
-   **Model Saving Logic:** The training script in `Q3/train.py` includes specific logic for saving the trained agent. It saves the agent as "best" whenever the current average return exceeds the previously recorded best score. Additionally, the script provides an option to save the current state of the agent as `sac_test_ckpt` if the training process is interrupted by a `KeyboardInterrupt`.
-   **Video Recording:** The training setup for the first environment in Q3 includes the `RecordVideo` wrapper from Gymnasium, which records videos of the agent's performance every 100 episodes in a "video" folder.
-   **Entropy Auto-tuning:** The SAC implementation includes entropy auto-tuning, which automatically adjusts the entropy regularization coefficient during training. This helps balance exploration and exploitation without requiring manual tuning of the entropy weight.

# 📊 Result

> SAC agent's outcome for the Humanoid Walk environment.

Best performance: 819.25 (std: 43.54)

![Training curves](assets\returns.png)

The score plunges at the 1500-th episode due to a rise of the entropy coefficient, attempting to do more exploration.

# Reference

- [Soft Actor-Critic (SAC) - CleanRL](https://docs.cleanrl.dev/rl-algorithms/sac/)
