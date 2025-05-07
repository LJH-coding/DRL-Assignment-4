import os
import sys
import matplotlib.pyplot as plt
import numpy as np

os.environ["MUJOCO_GL"] = "egl"  # Set up OpenGL context for video record
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

sys.path.append("..")  # access dmc in parent dir
from dmc import make_dmc_env

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import tensorclass

# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############
# Networks #
############


def init_weights(m):
    """Initialize weights orthogonally for better stability."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Qnet(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
        super(Qnet, self).__init__()
        n_observation = int(np.prod(envs.single_observation_space.shape))
        n_action = int(np.prod(envs.single_action_space.shape))

        self.fc = nn.Sequential(
            nn.Linear(n_observation + n_action, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.apply(init_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=1)
        return self.fc(state_action)


# reference: ClearRL
class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
        super().__init__()
        n_observation = int(np.prod(envs.single_observation_space.shape))
        n_action = int(np.prod(envs.single_action_space.shape))
        action_high = envs.single_action_space.high
        action_low = envs.single_action_space.low

        self.fc = nn.Sequential(
            nn.Linear(n_observation, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, n_action)
        self.fc_logstd = nn.Linear(256, n_action)  # log std is more stable
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

        self.apply(init_weights)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get mean, std
        x = self.fc(state)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )
        std = log_std.exp()

        # get action, log_prob
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)  # output between [-1, 1]
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean is deterministic action
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


#########
# Agent #
#########


@tensorclass
class Transition:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class SACAgent:
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        q_learning_rate: float = 1e-3,
        actor_learning_rate: float = 3e-4,
        replay_buffer_size: int = 20000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 5e-3,
        policy_update_freq: float = 2,
        target_update_freq: float = 2,
        use_jit: bool = True,
    ):
        self.use_jit = use_jit

        # networks
        self.actor = Actor(envs).to(device)
        self.q1 = Qnet(envs).to(device)
        self.q2 = Qnet(envs).to(device)
        self.target_q1 = Qnet(envs).to(device)
        self.target_q2 = Qnet(envs).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        if use_jit:
            self._make_jit_models(envs)
        self.target_q1.eval()
        self.target_q2.eval()

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_learning_rate
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )

        # automatic alpha tuning
        self.TARGET_ENTROPY = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=device
        )  # log is more stable
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=q_learning_rate)

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(replay_buffer_size, device=device),
            batch_size=batch_size,
            prefetch=10,
        )

    def _make_jit_models(self, envs: gym.vector.SyncVectorEnv) -> None:
        obs = envs.single_observation_space.sample()
        sample_input = torch.tensor([obs], device=device, dtype=torch.float32)
        self.actor = torch.jit.trace(self.actor, sample_input)
        self.q1 = torch.jit.script(self.q1)
        self.q2 = torch.jit.script(self.q2)
        self.target_q1 = torch.jit.script(self.target_q1)
        self.target_q2 = torch.jit.script(self.target_q2)

    def _update_q(self, batch: Transition) -> None:
        with torch.no_grad():
            sampled_action, log_prob, _ = self.actor(batch.next_obs)
            q_value_1 = self.target_q1(batch.next_obs, sampled_action)
            q_value_2 = self.target_q2(batch.next_obs, sampled_action)
            min_q_value = torch.min(q_value_1, q_value_2) - self.alpha * log_prob
            target_q_value = batch.reward + self.gamma * (1 - batch.done) * min_q_value

        q1_value = self.q1(batch.observation, batch.action)
        q2_value = self.q2(batch.observation, batch.action)
        q1_loss = F.mse_loss(q1_value, target_q_value)
        q2_loss = F.mse_loss(q2_value, target_q_value)
        total_q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

    def _update_actor(self, batch: Transition) -> None:
        sampled_action, log_prob, _ = self.actor(batch.observation)
        q_value_1 = self.q1(batch.observation, sampled_action)
        q_value_2 = self.q2(batch.observation, sampled_action)
        min_q_value = torch.min(q_value_1, q_value_2)  # detach?
        loss = (self.alpha * log_prob - min_q_value).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # automatic alpha tuning
        with torch.no_grad():
            _, log_prob, _ = self.actor(batch.observation)
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.TARGET_ENTROPY)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

    def _update_target_net(self) -> None:
        for param, target_param in zip(
            self.q1.parameters(), self.target_q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.q2.parameters(), self.target_q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self, global_step: int) -> None:
        batch = self.replay_buffer.sample()
        self._update_q(batch)
        if global_step % self.policy_update_freq == 0:
            for _ in range(self.policy_update_freq):
                self._update_actor(batch)
        if global_step % self.target_update_freq == 0:
            self._update_target_net()

    def get_action(self, observation: torch.Tensor):
        sampled_action, _, _ = self.actor(observation)
        return sampled_action.detach().cpu().numpy()

    def memorize(self, transition: Transition) -> None:
        self.replay_buffer.extend(transition)

    def save(self, dir_name: str) -> None:
        os.makedirs(dir_name, exist_ok=True)

        if self.use_jit:
            torch.jit.save(self.actor, f"{dir_name}/actor.pt")
            torch.jit.save(self.q1, f"{dir_name}/q1.pt")
            torch.jit.save(self.q2, f"{dir_name}/q2.pt")
        else:
            torch.save(self.actor.state_dict(), f"{dir_name}/actor.pt")
            torch.save(self.q1.state_dict(), f"{dir_name}/q1.pt")
            torch.save(self.q2.state_dict(), f"{dir_name}/q2.pt")

    def load(self, dir_name: str) -> None:
        if self.use_jit:
            self.actor = torch.jit.load(f"{dir_name}/actor.pt", map_location=device)
            self.q1 = torch.jit.load(f"{dir_name}/q1.pt", map_location=device)
            self.q2 = torch.jit.load(f"{dir_name}/q2.pt", map_location=device)
        else:
            self.actor.load_state_dict(
                torch.load(f"{dir_name}/actor.pt", map_location=device)
            )
            self.q1.load_state_dict(
                torch.load(f"{dir_name}/q1.pt", map_location=device)
            )
            self.q2.load_state_dict(
                torch.load(f"{dir_name}/q2.pt", map_location=device)
            )

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def eval(self) -> None:
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()

    def train(self) -> None:
        self.actor.train()
        self.q1.train()
        self.q2.train()


############
# Training #
############


def train(
    envs: gym.vector.SyncVectorEnv,
    agent: SACAgent,
    returns: list[float],
    total_steps: int = 5000000,
    warm_up_steps: int = 5000,
) -> None:
    observations, info = envs.reset()
    observations = torch.tensor(observations, device=device, dtype=torch.float32)
    for step in range(1, total_steps + 1):
        # select action and act
        if step < warm_up_steps:
            actions = envs.action_space.sample().astype(np.float32)
        else:
            actions = agent.get_action(observations)
        next_obss, rewards, terminated, truncated, info = envs.step(actions)
        next_obss = torch.tensor(next_obss, device=device, dtype=torch.float32)
        done = terminated | truncated

        # save transition to replay buffer
        transitions = Transition(
            observation=observations,
            action=torch.tensor(actions, dtype=torch.float32),
            reward=torch.tensor(rewards, dtype=torch.float32).unsqueeze_(-1),
            next_obs=next_obss,
            done=torch.tensor(terminated, dtype=torch.float32).unsqueeze_(-1),
            batch_size=[num_envs],
        )
        agent.memorize(transitions)

        # update models
        if step > warm_up_steps:
            agent.learn(step)

        if any(done):
            # log
            print(
                f'[{step}], alpha={agent.alpha:.3f}, returns={info["episode"]["r"]}, time={np.mean(info["episode"]["t"])}'
            )
            returns.append(np.average(info["episode"]["r"]))
            if (step % 100000) == 0:
                plt.clf()
                plt.plot(returns)
                plt.savefig("returns")
            # reset envs
            observations, info = envs.reset()
            observations = torch.tensor(
                observations, device=device, dtype=torch.float32
            )
            continue

        observations = next_obss


def make_env(idx: int):
    def thunk():
        # Create environment with state observations
        env_name = "humanoid-walk"
        env = make_dmc_env(
            env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False
        )
        # warppers
        env = RecordEpisodeStatistics(env, buffer_length=1000)
        if idx == 0:
            env = RecordVideo(
                env, video_folder="video", episode_trigger=lambda n: n % 100 == 0
            )
        return env

    return thunk


if __name__ == "__main__":
    returns = []
    num_envs = 4
    envs = gym.vector.SyncVectorEnv([make_env(idx) for idx in range(num_envs)])
    agent = SACAgent(
        envs,
        q_learning_rate=5e-4,
        actor_learning_rate=5e-4,
        replay_buffer_size=200000,
        batch_size=512,
        policy_update_freq=2,
        target_update_freq=2,
        use_jit=False,
    )

    print("[Action]")
    print(envs.single_action_space)
    print(envs.single_action_space.sample())
    print("-------------------------------------------")
    print("[Observation]")
    print(envs.single_observation_space)
    print(envs.single_observation_space.sample())

    try:
        train(envs, agent, returns)
    except KeyboardInterrupt:
        pass
    finally:
        envs.close()
        plt.clf()
        plt.plot(returns)
        plt.savefig("returns")
        agent.save("sac_test_ckpt")
