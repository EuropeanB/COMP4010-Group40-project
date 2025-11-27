import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from collections import deque

"""
Actor-Critic Agent, used to compare the performance with PPO agent
In actual training, PPO is preferred over AC 
For example, each update, AC uses only on-policy data once, while PPO reuses data multiple times with clipping to stabilize training.
Although the PPO agent trains slower per iteration, it achieves better sample efficiency and overall performance
As result, in AC training, we can see that the agent iteration speed is faster than PPO, but the training quality is worse
# gamma: discount factor
# entropy_coef: coefficient for entropy bonus
# value_coef: coefficient for value loss
# learning_rate: learning rate for optimizer
# rollout_steps: number of steps to collect per rollout
# max_iterations: total number of training iterations
"""

def AC(envs, actor_critic, save_path, device="cpu",
       gamma=0.99,
       entropy_coef=0.01,
       value_coef=0.5,
       learning_rate=2.5e-4,
       rollout_steps=32,
       max_iterations=200000):

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    # envir info
    num_envs = len(envs)
    channels, rows, cols = actor_critic.board_shape
    player_dim = actor_critic.player_dim
    action_size = actor_critic.num_actions

    # METRIC TRACKING 
    max_reward = -999999
    smoothed_rewards = deque(maxlen=2000)
    smoothing_factor = 0.9

    episode_rewards = np.zeros(num_envs)
    episode_rewards_list = deque(maxlen=2000)
    episode_lengths_list = deque(maxlen=2000)
    step_rewards_list = deque(maxlen=5000)

    first_move_orb_list = deque(maxlen=2000)
    perfect_level_ups_list = deque(maxlen=2000)
    decent_level_ups_list = deque(maxlen=2000)
    poor_level_ups_list = deque(maxlen=2000)

    entropy_values = deque(maxlen=2000)

    # Main training loop
    for iteration in tqdm(range(max_iterations)):

        # Rollout buffers
        board_buf = []
        player_buf = []
        action_buf = []
        reward_buf = []
        value_buf = []
        logprob_buf = []
        done_buf = []
        mask_buf = []

        # roll out
        for _ in range(rollout_steps):

            # Collect batched observations
            board_batch = torch.stack([
                torch.as_tensor(obs["board"], dtype=torch.float32)
                for obs in envs.observations
            ]).to(device)

            player_batch = torch.stack([
                torch.as_tensor(obs["player"], dtype=torch.float32)
                for obs in envs.observations
            ]).to(device)

            mask_batch = torch.stack([
                torch.as_tensor(obs["mask"], dtype=torch.bool)
                for obs in envs.observations
            ]).to(device)

            # Forward pass
            with torch.no_grad():
                logits, values = actor_critic(board_batch, player_batch)
                masked_logits = logits.masked_fill(~mask_batch, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)

                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            # Step environment
            rewards, dones = [], []
            for env_id, action in enumerate(actions):

                obs, r, terminated, truncated, info = envs.step(env_id, action.item())

                # Track rewards
                episode_rewards[env_id] += r
                step_rewards_list.append(r)

                # Track ORB on first move
                if envs.episode_steps[env_id] == 1:
                    first_move_orb_list.append(1 if info["last touched"] == "ORB" else 0)

                # Track level-up quality
                if info["levelled up"]:
                    if info["prev hp"] == 1:
                        perfect_level_ups_list.append(1)
                        decent_level_ups_list.append(0)
                        poor_level_ups_list.append(0)
                    elif info["prev hp"] == 2:
                        perfect_level_ups_list.append(0)
                        decent_level_ups_list.append(1)
                        poor_level_ups_list.append(0)
                    else:
                        perfect_level_ups_list.append(0)
                        decent_level_ups_list.append(0)
                        poor_level_ups_list.append(1)

                rewards.append(r)
                dones.append(terminated or truncated)

                # Episode finished
                if terminated or truncated:
                    episode_rewards_list.append(envs.total_rewards[env_id])
                    episode_lengths_list.append(envs.episode_steps[env_id])

                    # Smoothed reward
                    if len(smoothed_rewards) > 0:
                        smoothed_rewards.append(
                            smoothed_rewards[-1] * smoothing_factor +
                            (1 - smoothing_factor) * envs.total_rewards[env_id]
                        )
                    else:
                        smoothed_rewards.append(envs.total_rewards[env_id])

                    # Save best model
                    if envs.total_rewards[env_id] > max_reward:
                        max_reward = envs.total_rewards[env_id]
                        torch.save(actor_critic.state_dict(), f"{save_path}/best_ac.pth")

                    episode_rewards[env_id] = 0
                    envs.reset_env(env_id)

            # Save rollout step
            board_buf.append(board_batch)
            player_buf.append(player_batch)
            action_buf.append(actions)
            reward_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            value_buf.append(values.squeeze(-1))
            logprob_buf.append(logprobs)
            done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            mask_buf.append(mask_batch)

        # compute returns and advantages
        with torch.no_grad():
            board_batch = torch.stack([
                torch.as_tensor(obs["board"], dtype=torch.float32)
                for obs in envs.observations
            ]).to(device)

            player_batch = torch.stack([
                torch.as_tensor(obs["player"], dtype=torch.float32)
                for obs in envs.observations
            ]).to(device)

            _, last_values = actor_critic(board_batch, player_batch)
            last_values = last_values.squeeze(-1)

        advantages = torch.zeros(num_envs, device=device)
        return_buf = []

        for t in reversed(range(rollout_steps)):
            reward = reward_buf[t]
            done = done_buf[t]
            value = value_buf[t]

            next_value = last_values if t == rollout_steps - 1 else value_buf[t + 1]

            delta = reward + gamma * next_value * (1 - done) - value
            advantages = delta + gamma * advantages * (1 - done)

            return_buf.insert(0, advantages + value)

        # flatten buffers
        flat_board = torch.cat(board_buf, dim=0)
        flat_player = torch.cat(player_buf, dim=0)
        flat_actions = torch.cat(action_buf, dim=0)
        flat_logprobs_old = torch.cat(logprob_buf, dim=0)
        flat_returns = torch.cat(return_buf, dim=0)
        flat_values_old = torch.cat(value_buf, dim=0)
        flat_masks = torch.cat(mask_buf, dim=0)

        # update policy and value network
        logits, values = actor_critic(flat_board, flat_player)
        masked_logits = logits.masked_fill(~flat_masks, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)

        logprobs = dist.log_prob(flat_actions)
        entropy = dist.entropy().mean()
        entropy_values.append(entropy.item())

        advantages = flat_returns - values.squeeze(-1)

        policy_loss = -(advantages.detach() * logprobs).mean()
        value_loss = F.mse_loss(values.squeeze(-1), flat_returns)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # optimize
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        optimizer.step()

        #  print
        if iteration % 100 == 0 and iteration != 0:

            ep_window = 200
            step_window = 500

            avg_ep_reward = np.mean(list(episode_rewards_list)[-ep_window:]) if episode_rewards_list else 0
            avg_ep_length = np.mean(list(episode_lengths_list)[-ep_window:]) if episode_lengths_list else 0
            avg_step_reward = np.mean(list(step_rewards_list)[-step_window:]) if step_rewards_list else 0
            avg_entropy = np.mean(list(entropy_values)[-ep_window:]) if entropy_values else 0

            orb_rate = (sum(first_move_orb_list) / len(first_move_orb_list)) if len(first_move_orb_list) > 0 else 0
            perfect_rate = (sum(perfect_level_ups_list) / len(perfect_level_ups_list)) if len(perfect_level_ups_list) > 0 else 0
            decent_rate = (sum(decent_level_ups_list) / len(decent_level_ups_list)) if len(decent_level_ups_list) > 0 else 0
            poor_rate = (sum(poor_level_ups_list) / len(poor_level_ups_list)) if len(poor_level_ups_list) > 0 else 0

            smooth_reward = smoothed_rewards[-1] if smoothed_rewards else 0

            output = (
                f"\nIteration {iteration}"
                f"| Avg episode reward: {avg_ep_reward:.3f}"
                f" | Avg step reward: {avg_step_reward:.3f}"
                f" | Avg ep length: {avg_ep_length:.1f}"
                f" | Smoothed reward: {smooth_reward:.3f}"
                f" | Entropy: {avg_entropy:.3f}"
                f" | ORB First%: {orb_rate * 100:.1f}%"
                f" | Level Dist.: {perfect_rate * 100:.1f}/{decent_rate * 100:.1f}/{poor_rate * 100:.1f}\n"
            )

            print(output)
