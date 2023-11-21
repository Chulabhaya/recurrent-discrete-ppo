# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gym_gridverse
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment, GymStateWrapper
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import \
    make_observation_representation
from gym_gridverse.representations.state_representations import \
    make_state_representation


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="gridverse/gv_memory.7x7.yaml",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=20000000,
        help="total timesteps of the experiments")
    parser.add_argument("--maximum-episode-length", type=int, default=200,
        help="maximum length for episodes for gym POMDP environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, max_episode_len=None, mdp=False):
    def thunk():
        inner_env = factory_env_from_yaml(env_id)
        state_representation = make_state_representation(
            "compact",
            inner_env.state_space,
        )
        observation_representation = make_observation_representation(
            "compact",
            inner_env.observation_space,
        )
        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        if mdp:
            env = GymStateWrapper(GymEnvironment(outer_env))
        else:
            env = GymEnvironment(outer_env)

        if max_episode_len is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias'):
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
        self.conv1 = layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2))
        self.conv2 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2))

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
        self.conv1 = layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2))
        self.conv2 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2))

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


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, args.maximum_episode_length, mdp=True) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    actor = DiscreteActorGridVerseObs(envs).to(device)
    critic = DiscreteCriticGridVerseObs(envs).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = {
        "agent": torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.spaces["agent"].shape).to(device),
        "agent_id_grid": torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.spaces["agent_id_grid"].shape, dtype=torch.long).to(device),
        "grid": torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.spaces["grid"].shape, dtype=torch.long).to(device)
    }
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs["agent"] = torch.tensor(next_obs["agent"], dtype=torch.float32).to(device)
    next_obs["agent_id_grid"] = torch.tensor(next_obs["agent_id_grid"]).to(device)
    next_obs["grid"] = torch.tensor(next_obs["grid"]).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)
    next_truncated = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs["agent"][step] = next_obs["agent"]
            obs["agent_id_grid"][step] = next_obs["agent_id_grid"]
            obs["grid"][step] = next_obs["grid"]
            terminateds[step] = next_terminated
            truncateds[step] = next_truncated

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = actor.get_actions(next_obs)
                value = critic(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs["agent"] = torch.tensor(next_obs["agent"], dtype=torch.float32).to(device)
            next_obs["agent_id_grid"] = torch.tensor(next_obs["agent_id_grid"]).to(device)
            next_obs["grid"] = torch.tensor(next_obs["grid"]).to(device)
            next_terminated, next_truncated = torch.Tensor(terminated).to(device), torch.Tensor(truncated).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminateds[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.copy()
        b_obs["agent"] = b_obs["agent"].reshape((-1,) + envs.single_observation_space.spaces["agent"].shape)
        b_obs["agent_id_grid"] = b_obs["agent_id_grid"].reshape((-1,) + envs.single_observation_space.spaces["agent_id_grid"].shape)
        b_obs["grid"] = b_obs["grid"].reshape((-1,) + envs.single_observation_space.spaces["grid"].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {}
                mb_obs["agent"] = b_obs["agent"][mb_inds]
                mb_obs["agent_id_grid"] = b_obs["agent_id_grid"][mb_inds]
                mb_obs["grid"] = b_obs["grid"][mb_inds]

                _, newlogprob, entropy = actor.get_actions(mb_obs, b_actions.long()[mb_inds])
                newvalue = critic(mb_obs)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()