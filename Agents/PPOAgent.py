from torch import nn
from tqdm import tqdm
import torch
import gymnasium as gym
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super().__init__()

        self.num_actions = num_actions
        self.board_shape = board_shape
        self.player_dim = player_dim
        rows, cols, channels = self.board_shape

        # CNN for board processing
        self.board_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Calculate CNN output size
        self.conv_out_size = 64 * rows * cols

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + player_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Actor head (policy logits)
        self.actor = nn.Linear(256, num_actions)

        # Critic head (state value)
        self.critic = nn.Linear(256, 1)


    def forward(self, board_obs, player_obs):
        # board_obs: [batch, rows, cols, channels] -> [batch, channels, rows, cols]
        board_obs = board_obs.permute(0, 3, 1, 2)
        board_features = self.board_conv(board_obs)
        board_features = board_features.reshape(board_features.size(0), -1)

        combined = torch.cat([board_features, player_obs], dim=1)
        features = self.fc(combined)

        policy_logits = self.actor(features)
        state_value = self.critic(features)

        return policy_logits, state_value


class Environments:
    def __init__(self, num_actors):
        self.envs = [self.get_env() for _ in range(num_actors)]
        self.observations = [None for _ in range(num_actors)]
        self.done = [False for _ in range(num_actors)]
        self.total_rewards = [0 for _ in range(num_actors)]
        self.episode_steps  = [0 for _ in range(num_actors)]
        self.num_actors = num_actors

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
        return obs, info

    def step(self, env_id, action):
        observation, reward, terminated, truncated, info = self.envs[env_id].step(action)
        self.done[env_id] = terminated or truncated
        self.total_rewards[env_id] += reward
        self.episode_steps[env_id] += 1
        self.observations[env_id] = observation

        return observation, reward, terminated, truncated, info

    def get_env(self):
        env = gym.make("Dragonsweeper-v0", render_mode=None)
        return env


def PPO(envs, actor_critic, device='cpu'):
    rows, cols, channels = actor_critic.board_shape
    player_dim = actor_critic.player_dim

    # Hyperparameters
    T = 128 # Number of time steps collected per environment before performing an update
    K = 3 # Number of epochs per PPO update
    batch_size = 64
    gamma = 0.99
    gae_parameter = 0.95 # Generalized Advantage Estimation parameter
    vf_coef_c1 = 1 # Weight of the value loss in total PPO loss
    ent_coef_c2 = 0.01 # Weight of the entropy bonus in PPO loss
    num_iterations = 20_000

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=2.5e-4)

    # For tracking progress
    max_reward = 0
    episode_rewards = np.zeros(len(envs))
    total_rewards_list = []
    smoothed_rewards = []
    smoothing_factor = 0.9
    episode_lengths_list = []

    # Loading checkpoint if needed
    '''checkpoint = torch.load("Models/checkpoint_20.pth")
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])'''

    for iteration in tqdm(range(num_iterations)):
        advantages = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_board = torch.zeros((len(envs), T, rows, cols, channels), dtype=torch.float32, device=device)
        buffer_player = torch.zeros((len(envs), T, player_dim), dtype=torch.float32, device=device)
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
                m = torch.distributions.Categorical(logits=logits)
                actions = m.sample()
                log_probs = m.log_prob(actions)

            # Step environments
            next_obs, rewards, terms, truncs, infos = [], [], [], [], []
            for env_id, action in enumerate(actions):
                obs, reward, terminated, truncated, info = envs.step(env_id, action.item()) # why action.item() and not just "action"?
                episode_rewards[env_id] += reward

                # Append results
                next_obs.append(obs)
                rewards.append(reward)
                terms.append(terminated)
                truncs.append(truncated)
                infos.append(info)

                # Track rewards and best performing models
                if terminated or truncated:
                    total_rewards_list.append(envs.total_rewards[env_id])
                    episode_lengths_list.append(envs.episode_steps[env_id])

                    if smoothed_rewards:
                        smoothed_rewards.append(smoothing_factor * smoothed_rewards[-1] + (1 - smoothing_factor) * envs.total_rewards[env_id])
                    else:
                        smoothed_rewards.append(envs.total_rewards[env_id])

                    if envs.total_rewards[env_id] > max_reward:
                        max_reward = envs.total_rewards[env_id]
                        torch.save(actor_critic.state_dict(), f"/Users/arthurteixeira/Desktop/Pycharm/Latest/Models/best_agent.pth")

                    # Reset the environment
                    episode_rewards[env_id] = 0
                    envs.reset_env(env_id)

            # Log into buffers
            buffer_board[:, t] = board_batch
            buffer_player[:, t] = player_batch
            buffer_actions[:, t] = actions
            buffer_logprobs[:, t] = log_probs
            buffer_values[:, t] = values.squeeze(-1)
            buffer_rewards[:, t] = torch.tensor(rewards, device=device, dtype=torch.float32)
            buffer_dones[:, t] = torch.tensor([t_ or f_ for t_, f_ in zip(terms, truncs)], device=device, dtype=torch.float32)

        # Compute value for the last step
        board_batch = torch.stack([torch.as_tensor(obs['board'], dtype=torch.float32) for obs in envs.observations]).to(device)
        player_batch = torch.stack([torch.as_tensor(obs['player'], dtype=torch.float32) for obs in envs.observations]).to(device)

        # Forward pass
        with torch.no_grad():
            _, last_values = actor_critic(board_batch, player_batch)
        buffer_values[:, T] = last_values.squeeze(-1)

        # Compute GAE advantages
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - buffer_dones[:, t]
            delta = buffer_rewards[:, t] + gamma * buffer_values[:, t+1] * next_non_terminal - buffer_values[:, t]
            if t == (T - 1):
                advantages[:, t] = delta
            else:
                advantages[:, t] = delta + gamma * gae_parameter * advantages[:, t+1] * next_non_terminal

        # Flatten for training
        flat_board = buffer_board.reshape(-1, rows, cols, channels)
        flat_player = buffer_player.reshape(-1, player_dim)
        flat_actions = buffer_actions.reshape(-1)
        flat_old_logprobs = buffer_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8) # Added for testing
        flat_returns = (advantages + buffer_values[:, :T]).reshape(-1)
        flat_old_values = buffer_values[:, :T].reshape(-1)

        # Create dataset and loader for PPO update
        dataset = TensorDataset(flat_advantages, flat_board, flat_player, flat_actions, flat_old_logprobs, flat_returns, flat_old_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PPO update
        for _ in range(K):
            for b_adv, b_board, b_player, b_actions, b_logprob_old, b_returns, b_old_values in loader:
                logits, values = actor_critic(b_board, b_player)
                m = torch.distributions.Categorical(logits=logits)
                log_probs = m.log_prob(b_actions)
                ratio = torch.exp(log_probs - b_logprob_old)
                policy_loss_1 = b_adv * ratio
                #clip_range =  0.1 * (1.0 - iteration / num_iterations)
                clip_range = 0.1
                policy_loss_2 = b_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Clipped value loss
                value_pred_clipped = b_old_values + torch.clamp(values - b_old_values, -clip_range, +clip_range)
                value_loss_unclipped = (values - b_returns) ** 2
                value_loss_clipped = (value_pred_clipped - b_returns) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Compute total loss
                loss = policy_loss + ent_coef_c2 * -m.entropy().mean() + vf_coef_c1 * value_loss

                # Clip the gradient and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()

        # Log reward
        if iteration % 10 == 0:
            avg_reward = np.mean(total_rewards_list[-len(envs):]) if total_rewards_list else 0
            smooth_reward = smoothed_rewards[-1] if smoothed_rewards else 0
            avg_length = np.mean(episode_lengths_list[-100:]) if episode_lengths_list else 0
            print(f"\nIteration {iteration} | Avg reward (recent episodes): {avg_reward:.2f} | Smoothed reward: {smooth_reward:.2f} | Avg. episode length: {avg_length:.1f}")

        if iteration % 500 == 0 and iteration != 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': actor_critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, f"/Users/arthurteixeira/Desktop/Pycharm/Latest/Models/checkpoint_{iteration}.pth")