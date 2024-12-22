# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from common.models import DiscreteActorGridVerseObs, DiscreteCriticGridVerseObs
from common.utils import make_env, save, set_seed


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-group", type=str, default="memory_four_rooms",
        help="the group under which this experiment falls")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project", type=str, default="ppo_gv_mdp_baselines",
        help="wandb project name")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="gridverse/gv_memory_four_rooms.7x7.yaml",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--maximum-episode-length", type=int, default=100,
        help="maximum length for episodes for gym POMDP environment")
    parser.add_argument("--learning-rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
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

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=100000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default="trained_models/ppo_gridverse_obs_discrete_action_wkiznsen/global_step_40000000.pth",
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default=None,
        help="wandb unique run id for resuming")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        run_id = args.run_id
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            resume="must",
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            mode="online",
        )

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeding
    set_seed(args.seed, device)

    # Load checkpoint if resuming
    if args.resume:
        print("Resuming from checkpoint: " + args.resume_checkpoint_path, flush=True)
        checkpoint = torch.load(args.resume_checkpoint_path)

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, args.maximum_episode_length, mdp=True)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Initialize models and optimizers
    actor = DiscreteActorGridVerseObs(envs).to(device)
    critic = DiscreteCriticGridVerseObs(envs).to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # If resuming training, load models and optimizers
    if args.resume:
        actor.load_state_dict(checkpoint["model_state_dict"]["actor_state_dict"])
        critic.load_state_dict(checkpoint["model_state_dict"]["critic_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["optimizer"])

    # ALGO Logic: Storage setup
    obs = {
        "agent": torch.zeros(
            (args.num_steps, args.num_envs)
            + envs.single_observation_space.spaces["agent"].shape
        ).to(device),
        "agent_id_grid": torch.zeros(
            (args.num_steps, args.num_envs)
            + envs.single_observation_space.spaces["agent_id_grid"].shape,
            dtype=torch.long,
        ).to(device),
        "grid": torch.zeros(
            (args.num_steps, args.num_envs)
            + envs.single_observation_space.spaces["grid"].shape,
            dtype=torch.long,
        ).to(device),
    }
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncateds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs["agent"] = torch.tensor(next_obs["agent"], dtype=torch.float32).to(device)
    next_obs["agent_id_grid"] = torch.tensor(next_obs["agent_id_grid"]).to(device)
    next_obs["grid"] = torch.tensor(next_obs["grid"]).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)
    next_truncated = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Store values for data logging
        data_log = {}

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
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs["agent"] = torch.tensor(next_obs["agent"], dtype=torch.float32).to(
                device
            )
            next_obs["agent_id_grid"] = torch.tensor(next_obs["agent_id_grid"]).to(
                device
            )
            next_obs["grid"] = torch.tensor(next_obs["grid"]).to(device)
            next_terminated, next_truncated = (
                torch.Tensor(terminated).to(device),
                torch.Tensor(truncated).to(device),
            )

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                data_log["misc/episodic_return"] = info["episode"]["r"]
                data_log["misc/episodic_length"] = info["episode"]["l"]

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
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.copy()
        b_obs["agent"] = b_obs["agent"].reshape(
            (-1,) + envs.single_observation_space.spaces["agent"].shape
        )
        b_obs["agent_id_grid"] = b_obs["agent_id_grid"].reshape(
            (-1,) + envs.single_observation_space.spaces["agent_id_grid"].shape
        )
        b_obs["grid"] = b_obs["grid"].reshape(
            (-1,) + envs.single_observation_space.spaces["grid"].shape
        )
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

                _, newlogprob, entropy = actor.get_actions(
                    mb_obs, b_actions.long()[mb_inds]
                )
                newvalue = critic(mb_obs)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
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
        data_log["misc/learning_rate"] = optimizer.param_groups[0]["lr"]
        data_log["losses/value_loss"] = v_loss.item()
        data_log["losses/policy_loss"] = pg_loss.item()
        data_log["losses/entropy"] = entropy_loss.item()
        data_log["losses/old_approx_kl"] = old_approx_kl.item()
        data_log["losses/approx_kl"] = approx_kl.item()
        data_log["losses/clipfrac"] = np.mean(clipfracs)
        data_log["losses/explained_variance"] = explained_var
        data_log["misc/steps_per_second"] = int(
            global_step / (time.time() - start_time)
        )
        print("SPS:", int(global_step / (time.time() - start_time)))

        data_log["misc/global_step"] = global_step
        wandb.log(data_log, step=global_step)

        # Save checkpoints during training
        if args.save:
            if (global_step) % (args.checkpoint_interval / args.num_envs) == 0:
                # Save models
                models = {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                }
                # Save optimizers
                optimizers = {
                    "optimizer": optimizer.state_dict(),
                }

                save(
                    run_id,
                    args.save_checkpoint_dir,
                    global_step,
                    models,
                    optimizers,
                )

    envs.close()
