import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias"):
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActorGridVerseObs(nn.Module):
    """Discrete actor model for discrete SAC with discrete actions
    and GridVerse observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()

        # Process image and direction
        self.grid_embedding = layer_init(nn.Embedding(32, 4))
        self.agent_id_grid_embedding = layer_init(nn.Embedding(2, 4))
        self.conv1 = layer_init(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
        )
        self.conv2 = layer_init(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        )

        # Remainder of network
        self.fc1 = layer_init(nn.Linear(1606, 256))
        self.fc2 = layer_init(nn.Linear(256, env.single_action_space.n), std=0.01)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        # Generate embeddings
        grid_emb = self.grid_embedding(states["grid"])
        grid_emb = torch.flatten(grid_emb, start_dim=3)
        agent_id_grid_emb = self.agent_id_grid_embedding(states["agent_id_grid"])
        unified_grid_emb = torch.cat((grid_emb, agent_id_grid_emb), dim=3).permute(
            0, 3, 1, 2
        )
        unified_grid_emb = F.relu(self.conv1(unified_grid_emb))
        unified_grid_emb = F.relu(self.conv2(unified_grid_emb))
        unified_grid_emb = torch.flatten(unified_grid_emb, start_dim=1)
        agent_emb = states["agent"]

        # Process embeddings with FC layers
        x = torch.cat((unified_grid_emb, agent_emb), dim=1)
        x = F.relu(self.fc1(x))

        # Rest of the network
        action_logits = self.fc2(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states, actions=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Logs of action probabilities, used for entropy.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        if actions is None:
            actions = dist.sample().to(states["grid"].device)

        return actions, dist.log_prob(actions), dist.entropy()


class DiscreteCriticGridVerseObs(nn.Module):
    """Discrete soft Q-network model for discrete SAC with discrete actions
    and GridVerse observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()

        # Process image and direction
        self.grid_embedding = layer_init(nn.Embedding(32, 4))
        self.agent_id_grid_embedding = layer_init(nn.Embedding(2, 4))
        self.conv1 = layer_init(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
        )
        self.conv2 = layer_init(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        )

        # Remainder of network
        self.fc1 = layer_init(nn.Linear(1606, 256))
        self.fc2 = layer_init(nn.Linear(256, 1), std=1.0)

    def forward(self, states):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        # Generate embeddings
        grid_emb = self.grid_embedding(states["grid"])
        grid_emb = torch.flatten(grid_emb, start_dim=3)
        agent_id_grid_emb = self.agent_id_grid_embedding(states["agent_id_grid"])
        unified_grid_emb = torch.cat((grid_emb, agent_id_grid_emb), dim=3).permute(
            0, 3, 1, 2
        )
        unified_grid_emb = F.relu(self.conv1(unified_grid_emb))
        unified_grid_emb = F.relu(self.conv2(unified_grid_emb))
        unified_grid_emb = torch.flatten(unified_grid_emb, start_dim=1)
        agent_emb = states["agent"]

        # Process embeddings with FC layers
        x = torch.cat((unified_grid_emb, agent_emb), dim=1)
        x = F.relu(self.fc1(x))

        # Rest of the network
        q_values = self.fc2(x)

        return q_values
