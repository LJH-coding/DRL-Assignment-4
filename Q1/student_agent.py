import numpy as np
import torch

device = "cpu"


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.actor = torch.jit.load(f"actor.pt", map_location=device)
        self.actor.eval()

    def act(self, observation):
        observation = torch.tensor(
            np.array([observation]), dtype=torch.float32, device=device
        )
        _, _, deterministic_action = self.actor(observation)
        return deterministic_action.detach().cpu().numpy().squeeze(1)
