from torch import nn
from tqdm import tqdm
import torch
import gymnasium as gym
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super().__init__()

        # These variables are required for checks elsewhere
        self.board_shape = board_shape
        self.player_dim = player_dim
        self.num_actions = num_actions

        channels, rows, cols = board_shape
        self.num_cells = rows * cols

        # Per-cell encoder (local spatial decisions)
        # Input: cell features (5) + player state (2) = 7
        self.cell_encoder = nn.Sequential(
            nn.Linear(channels + player_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Board aggregator (global state representation)
        # Summarizes all cell embeddings into board-level features
        self.board_aggregator = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Cell action head: simple linear
        self.cell_action_head = nn.Linear(32, 1)

        # Level-up head: sees player state + aggregated board state
        self.level_up_head = nn.Sequential(
            nn.Linear(player_dim + 32, 64),  # player (2) + board_summary (32)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(32 + player_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, board_obs, player_obs):
        batch, channels, rows, cols = board_obs.shape # B< C< R, W
        num_cells = rows * cols

        # Encode each cell with player context
        # Flatten spatial dimensions
        cells = board_obs.permute(0, 2, 3, 1).reshape(batch * num_cells, channels)  # [B * 130, 5]

        # Broadcast player state to each cell
        player_expanded = player_obs.unsqueeze(1).expand(batch, num_cells, self.player_dim)
        player_expanded = player_expanded.reshape(batch * num_cells, self.player_dim)  # [B * 130, 2]

        # Concatenate so each cell sees its features and player state
        cell_input = torch.cat([cells, player_expanded], dim=1)  # [B * 130, 7]

        # Encode
        cell_emb = self.cell_encoder(cell_input)  # [B * 130, 32]
        cell_emb = cell_emb.reshape(batch, num_cells, 32)  # [B, 130, 32]

        # Compute cell action logits
        cell_logits = self.cell_action_head(cell_emb).squeeze(-1)  # [B, 130]

        # Aggregate board state for global decision
        # Mean pooling over all cells
        board_summary = cell_emb.mean(dim=1)  # [B, 32]
        board_summary = self.board_aggregator(board_summary)  # [B, 32]

        # Compute level-up logit
        # Level-up sees player state and summary of what's available on the board
        level_input = torch.cat([player_obs, board_summary], dim=1)  # [B, 34]
        level_logit = self.level_up_head(level_input)  # [B, 1]

        # Combine all action logits
        logits = torch.cat([cell_logits, level_logit], dim=1)  # [B, 131]

        # Value function
        critic_input = torch.cat([board_summary, player_obs], dim=1)  # [B, 34]
        values = self.critic(critic_input).squeeze(-1)  # [B]

        return logits, values


class Environments:
    def __init__(self, num_actors):
        self.envs = [self.get_env() for _ in range(num_actors)]
        self.observations = [None for _ in range(num_actors)]
        self.done = [False for _ in range(num_actors)]
        self.total_rewards = [0 for _ in range(num_actors)]
        self.episode_steps  = [0 for _ in range(num_actors)]
        self.num_actors = num_actors
        self.first_actor_clicked = [None for _ in range(num_actors)]

        for env_id in range(num_actors):
            self.reset_env(env_id)

    def __len__(self):
        return self.num_actors

    def reset_env(self, env_id):
        self.total_rewards[env_id] = 0
        self.episode_steps[env_id] = 0
        obs, info = self.envs[env_id].reset()
        self.observations[env_id] = obs
        self.done[env_id] = False
        self.first_actor_clicked[env_id] = None

        return obs, info

    def step(self, env_id, action):
        observation, reward, terminated, truncated, info = self.envs[env_id].step(action)
        self.done[env_id] = terminated or truncated
        self.total_rewards[env_id] += reward
        self.episode_steps[env_id] += 1
        self.observations[env_id] = observation

        if self.episode_steps[env_id] == 1:
            self.first_actor_clicked[env_id] = info['last touched']

        return observation, reward, terminated, truncated, info

    @staticmethod
    def get_env():
        env = gym.make("Dragonsweeper-v0", render_mode=None)
        return env


def PPO(envs, test_env, actor_critic, save_path, device='cpu'):
    channels, rows, cols = actor_critic.board_shape
    num_actions = rows * cols + 1
    player_dim = actor_critic.player_dim

    # Hyperparameters
    T = 128 # Number of time steps collected per environment before performing an update
    K = 4 # Number of epochs per PPO update
    batch_size = 128
    gamma = 0.99
    gae_parameter = 0.95 # Generalized Advantage Estimation parameter
    vf_coef_c1 = 0.5  # Weight of the value loss in total PPO loss
    ent_coef_c2 = 0.03 # Weight of the entropy bonus in PPO loss
    num_iterations = 1_000_000
    learning_rate = 1e-4

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    # For tracking progress
    max_reward = -10000
    episode_rewards = np.zeros(len(envs))

    # ---- METRIC TRACKING ---- #
    smoothed_rewards = deque(maxlen=2000)
    smoothing_factor = 0.9
    episode_rewards_list = deque(maxlen=2000)
    episode_lengths_list = deque(maxlen=2000)
    step_rewards_list = deque(maxlen=5000)
    first_move_orb_list = deque(maxlen=2000)
    perfect_level_ups_list = deque(maxlen=2000)
    decent_level_ups_list = deque(maxlen=2000)
    poor_level_ups_list = deque(maxlen=2000)
    illegal_actions_count = 0
    entropy_values = deque(maxlen=2000)
    # ------------------------- #


    # Loading checkpoint if needed
    '''checkpoint = torch.load("Models/best_agent.pth")
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])'''

    for iteration in tqdm(range(num_iterations)):
        buffer_board = torch.zeros((len(envs), T, channels, rows, cols), dtype=torch.float32, device=device)
        buffer_player = torch.zeros((len(envs), T, player_dim), dtype=torch.float32, device=device)
        buffer_masks = torch.zeros((len(envs), T, num_actions), dtype=torch.bool, device=device)
        buffer_actions = torch.zeros((len(envs), T), dtype=torch.long, device=device)
        buffer_logprobs = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_values = torch.zeros((len(envs), T+1), dtype=torch.float32, device=device)
        buffer_rewards = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_dones = torch.zeros((len(envs), T), dtype=torch.float32, device=device)

        for t in range(T):
            # Prepare batch of observations
            board_batch = torch.stack([torch.as_tensor(obs['board'], dtype=torch.float32) for obs in envs.observations]).to(device)
            player_batch = torch.stack([torch.as_tensor(obs['player'], dtype=torch.float32) for obs in envs.observations]).to(device)

            # Forward pass
            with torch.no_grad():
                logits, values = actor_critic(board_batch, player_batch)

                # Get the mask for each environment
                mask_batch = torch.stack([
                    torch.as_tensor(obs['mask'], dtype=torch.bool) for obs in envs.observations
                ]).to(device)

                # Set logits of invalid actions to a very large negative number
                masked_logits = logits.clone()
                masked_logits = masked_logits.masked_fill(~mask_batch, -1e9)

                # Sample using masked logits
                m = torch.distributions.Categorical(logits=masked_logits)
                actions = m.sample()
                log_probs = m.log_prob(actions)

            # Step environments
            rewards, dones = [], []
            for env_id, action in enumerate(actions):
                obs, reward, terminated, truncated, info = envs.step(env_id, action.item())

                # Log if illegal action was taken (mainly for testing)
                if not mask_batch[env_id, action.item()]:
                    illegal_actions_count += 1

                # Update logging rewards
                episode_rewards[env_id] += reward
                step_rewards_list.append(reward)
                if info['levelled up']:
                    if info['prev hp'] == 1:
                        perfect_level_ups_list.append(1.0)
                        decent_level_ups_list.append(0.0)
                        poor_level_ups_list.append(0.0)
                    elif info['prev hp'] == 2:
                        perfect_level_ups_list.append(0.0)
                        decent_level_ups_list.append(1.0)
                        poor_level_ups_list.append(0.0)
                    else:
                        perfect_level_ups_list.append(0.0)
                        decent_level_ups_list.append(0.0)
                        poor_level_ups_list.append(1.0)

                # Append results
                rewards.append(reward)
                dones.append(terminated or truncated)

                # Track rewards and best performing models
                if terminated or truncated:
                    episode_rewards_list.append(envs.total_rewards[env_id])
                    episode_lengths_list.append(envs.episode_steps[env_id])

                    if len(smoothed_rewards) > 0:
                        smoothed_rewards.append(smoothing_factor * smoothed_rewards[-1] + (1 - smoothing_factor) * envs.total_rewards[env_id])
                    else:
                        smoothed_rewards.append(envs.total_rewards[env_id])

                    first_move_orb_list.append(1 if envs.first_actor_clicked[env_id] == "ORB" else 0)

                    # Save best agent
                    if envs.total_rewards[env_id] > max_reward:
                        max_reward = envs.total_rewards[env_id]
                        torch.save(actor_critic.state_dict(), f"{save_path}/best_agent.pth")

                    # Reset the environment
                    episode_rewards[env_id] = 0
                    envs.reset_env(env_id)

            # Log into buffers
            buffer_board[:, t] = board_batch
            buffer_player[:, t] = player_batch
            buffer_actions[:, t] = actions
            buffer_logprobs[:, t] = log_probs
            buffer_values[:, t] = values
            buffer_rewards[:, t] = torch.tensor(rewards, device=device, dtype=torch.float32)
            buffer_dones[:, t] = torch.tensor(dones, device=device, dtype=torch.float32)
            buffer_masks[:, t] = mask_batch

        # Forward pass
        with torch.no_grad():
            board_batch = torch.stack([torch.as_tensor(obs['board'], dtype=torch.float32) for obs in envs.observations]).to(device)
            player_batch = torch.stack([torch.as_tensor(obs['player'], dtype=torch.float32) for obs in envs.observations]).to(device)
            _, last_values = actor_critic(board_batch, player_batch)
        buffer_values[:, T] = last_values

        # Compute GAE advantages
        advantages = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - buffer_dones[:, t]
            delta = buffer_rewards[:, t] + gamma * buffer_values[:, t+1] * next_non_terminal - buffer_values[:, t]
            if t == (T - 1):
                advantages[:, t] = delta
            else:
                advantages[:, t] = delta + gamma * gae_parameter * advantages[:, t+1] * next_non_terminal

        # Flatten for training
        flat_board = buffer_board.reshape(-1, channels, rows, cols)
        flat_player = buffer_player.reshape(-1, player_dim)
        flat_actions = buffer_actions.reshape(-1)
        flat_old_logprobs = buffer_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        flat_returns = (advantages + buffer_values[:, :T]).reshape(-1)
        flat_old_values = buffer_values[:, :T].reshape(-1).detach()
        flat_masks = buffer_masks.reshape(-1, actor_critic.num_actions)

        # Create dataset and loader for PPO update
        dataset = TensorDataset(flat_advantages, flat_board, flat_player, flat_actions, flat_old_logprobs, flat_returns, flat_old_values, flat_masks)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PPO update
        for _ in range(K):
            for b_adv, b_board, b_player, b_actions, b_logprob_old, b_returns, b_old_values, b_masks in loader:
                logits, values = actor_critic(b_board, b_player)
                masked_logits = logits + (~b_masks) * (-1e9)
                m = torch.distributions.Categorical(logits=masked_logits)
                log_probs = m.log_prob(b_actions)
                entropy = m.entropy()
                entropy_values.append(entropy.mean().item())

                ratio = torch.exp(log_probs - b_logprob_old)
                policy_loss_1 = b_adv * ratio
                clip_range = 0.2
                policy_loss_2 = b_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Clipped value loss
                value_pred_clipped = b_old_values + torch.clamp(values - b_old_values, -clip_range, +clip_range)
                value_loss_unclipped = (values - b_returns) ** 2
                value_loss_clipped = (value_pred_clipped - b_returns) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Compute total loss
                loss = policy_loss + ent_coef_c2 * -entropy.mean() + vf_coef_c1 * value_loss

                # Clip the gradient and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()

        # Log reward
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
            poor_rate = (sum(poor_level_ups_list) / len(poor_level_ups_list))  if len(perfect_level_ups_list) > 0 else 0

            smooth_reward = smoothed_rewards[-1] if smoothed_rewards else 0

            output = ""
            output += (f"\nIteration {iteration}"
                       f"| Avg episode reward: {avg_ep_reward:.3f}"
                       f" | Avg step reward: {avg_step_reward:.3f}"
                       f" | Avg ep length: {avg_ep_length:.1f}"
                       f" | Smoothed reward: {smooth_reward:.3f}"
                       f" | Entropy: {avg_entropy:.3f}"
                       f" | ORB First%: {orb_rate * 100:.1f}%"
                       f" | Level Dist.: {perfect_rate * 100:.1f}/{decent_rate * 100:.1f}/{poor_rate * 100:.1f}\n"
            )

            '''# Run test
            avg_test_ep_length = []
            avg_test_ep_reward = []
            avg_test_step_reward = []
            for _ in range(50):
                test_obs, _ = test_env.reset()
                test_step = 0
                test_total_reward = 0
                test_terminated = test_truncated = False
                while not (test_terminated or test_truncated):
                    test_board_obs = torch.as_tensor(test_obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
                    test_player_obs = torch.as_tensor(test_obs['player'], dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        logits, _ = actor_critic(test_board_obs, test_player_obs)
                        mask = torch.as_tensor(test_obs['mask'], dtype=torch.float32).to(device)
                        masked_logits = logits + (mask - 1) * 1e9
                        test_action = torch.argmax(masked_logits, dim=-1).item()
                    test_obs, test_reward, test_terminated, test_truncated, _ = test_env.step(test_action)
                    test_step += 1
                    test_total_reward += test_reward
                    avg_test_step_reward.append(test_reward)
                avg_test_ep_length.append(test_step)
                avg_test_ep_reward.append(test_total_reward)

            output += (
                    f"TESTING (50 Episodes)"
                    f" | Avg test episode reward: {np.mean(avg_test_ep_reward):.3f}"
                    f" | Avg test step reward: {np.mean(avg_test_step_reward):.3f}"
                    f" | Avg test ep length: {np.mean(avg_test_ep_length):.3f}"
            )'''
            print(output)

        if iteration % 500 == 0 and iteration != 0:
            torch.save(actor_critic.state_dict(), f"{save_path}/checkpoint_{iteration}.pth")