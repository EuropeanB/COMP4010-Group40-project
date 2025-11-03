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
        self.num_actors = num_actors

        for env_id in range(num_actors):
            self.reset_env(env_id)

    def __len__(self):
        return self.num_actors

    def reset_env(self, env_id):
        self.total_rewards[env_id] = 0
        obs, info = self.envs[env_id].reset()
        self.observations[env_id] = obs
        self.done[env_id] = False
        return obs, info

    def step(self, env_id, action):
        observation, reward, terminated, truncated, info = self.envs[env_id].step(action)
        self.done[env_id] = terminated or truncated
        self.total_rewards[env_id] += reward
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
    num_iterations = 100#40_000

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=2.5e-4)

    # For tracking progress
    max_reward = 0
    total_rewards = [[] for _ in range(len(envs))]
    smoothed_rewards = [[] for _ in range(len(envs))]

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

                # Append results
                next_obs.append(obs)
                rewards.append(reward)
                terms.append(terminated)
                truncs.append(truncated)
                infos.append(info)

                # Track max reward
                if terminated and envs.total_rewards[env_id] > max_reward:
                    max_reward = envs.total_rewards[env_id]
                    torch.save(actor_critic.cpu(), "best_agent")
                    actor_critic.to(device)

                # Reset the environment if it is done
                if terminated or truncated:
                    total_rewards[env_id].append(envs.total_rewards[env_id])
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
        flat_returns = (advantages + buffer_values[:, :T]).reshape(-1)

        # Create dataset and loader for PPO update
        dataset = TensorDataset(flat_advantages, flat_board, flat_player, flat_actions, flat_old_logprobs, flat_returns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PPO update
        for _ in range(K):
            for b_adv, b_board, b_player, b_actions, b_logprob_old, b_returns in loader:
                logits, values = actor_critic(b_board, b_player)
                m = torch.distributions.Categorical(logits=logits)
                log_probs = m.log_prob(b_actions)
                ratio = torch.exp(log_probs - b_logprob_old)
                policy_loss_1 = b_adv * ratio
                clip_range =  0.1 * (1.0 - iteration / num_iterations)
                policy_loss_2 = b_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Clipped value loss
                value_loss_1 = F.mse_loss(b_returns, values.squeeze(-1), reduction='none')
                value_loss_2 = F.mse_loss(b_returns, torch.clamp(values.squeeze(-1), values.squeeze(-1) - clip_range, values.squeeze(-1) + clip_range), reduction='none')
                value_loss = torch.max(value_loss_1, value_loss_2).mean()

                # Compute total loss
                loss = policy_loss + ent_coef_c2 * -m.entropy().mean() + vf_coef_c1 * value_loss

                # Clip the gradient and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()


### NON VECTORIZED LOOP ###
'''
    for iteration in tqdm(range(num_iterations)):
        advantages = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_board_states = torch.zeros((len(envs), T, rows, cols, channels), dtype=torch.float32, device=device)
        buffer_player_states = torch.zeros((len(envs), T, player_dim), dtype=torch.float32, device=device)
        buffer_actions = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_logprobs = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_state_values = torch.zeros((len(envs), T+1), dtype=torch.float32, device=device)
        buffer_rewards = torch.zeros((len(envs), T), dtype=torch.float32, device=device)
        buffer_is_terminal = torch.zeros((len(envs), T), dtype=torch.float32, device=device)

        # This can be done in parallel and will be much more efficient
        for env_id in range(len(envs)):
            with torch.no_grad():
                # Calculate values for time steps 0 to T-1
                for t in range(T):
                    obs = envs.observations[env_id]
                    board_obs = torch.as_tensor(obs['board'], device=device).unsqueeze(0)
                    player_obs = torch.as_tensor(obs['player'], device=device).unsqueeze(0)
                    logits, value = actor_critic(board_obs, player_obs)
                    logits, value = logits.squeeze(0), value.squeeze(0)
                    m = torch.distributions.categorical.Categorical(logits=logits)

                    if envs.done[env_id]:
                        action = torch.tensor([1], device=device)
                    else:
                        action = m.sample()

                    log_prob = m.log_prob(action)
                    observation, reward, terminated, truncated, info = envs.step(env_id, action)
                    reward = np.sign(reward) # Reward clipping (not sure how this applies in our case)

                    # Log everything
                    buffer_board_states[env_id, t] = board_obs
                    buffer_player_states[env_id, t] = player_obs
                    buffer_actions[env_id, t] = torch.as_tensor([action], device=device)
                    buffer_logprobs[env_id, t] = log_prob
                    buffer_state_values[env_id, t] = value
                    buffer_rewards[env_id, t] = reward
                    buffer_is_terminal[env_id, t] = (terminated or truncated)

                    # If model's game ends
                    if terminated:
                        # Save model if best performer so far
                        if envs.total_rewards[env_id] > max_reward:
                            max_reward = envs.total_rewards[env_id]
                            torch.save(actor_critic.cpu(), "best_agent")
                            actor_critic.to(device)

                        # Log total reward and reset environment
                        total_rewards[env_id].append(envs.total_rewards[env_id])
                        envs.reset_env(env_id)

                # Calculate value a time step T
                obs = envs.observations[env_id]
                board_obs = torch.as_tensor(obs['board'], device=device).unsqueeze(0)
                player_obs = torch.as_tensor(obs['player'], device=device).unsqueeze(0)
                values = actor_critic(board_obs, player_obs)[1].squeeze(0)
                buffer_state_values[env_id, T] = values

                # Compute advantage estimates A^1, ... , A^
                for t in range(T-1, -1, -1):
                    next_non_terminal = 1.0 - buffer_is_terminal[env_id, t]
                    delta_t = buffer_rewards[env_id, t] + gamma * buffer_state_values[env_id, t+1] * next_non_terminal - buffer_state_values[env_id, t]
                    if t == (T-1):
                        A_t = delta_t
                    else:
                        A_t = delta_t + gamma * gae_parameter * advantages[env_id, t+1] * next_non_terminal
                    advantages[env_id, t] = A_t

        advantages_data_loader = DataLoader(
            TensorDataset(
                advantages.reshape(advantages.shape[0] * advantages.shape[1]),
                buffer_board_states.reshape(-1, rows, cols, channels),
                buffer_player_states.reshape(-1, player_dim),
                buffer_actions.reshape(-1),
                buffer_logprobs.reshape(-1),
                buffer_state_values[:, :T].reshape(-1)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in range(K):
            for batch_advantages in advantages_data_loader:
                b_adv, board_obs, player_obs, action_that_was_taken, old_log_prob, old_state_values = batch_advantages

                logits, value = actor_critic(board_obs, player_obs)
                logits, value = logits.squeeze(0), value.squeeze(-1)
                m = torch.distributions.categorical.Categorical(logits=logits)
                log_prob = m.log_prob(action_that_was_taken)
                ratio = torch.exp(log_prob - old_log_prob)
                returns = b_adv + old_state_values

                # Clipped surrogate objective
                policy_loss_1 = b_adv * ratio
                alpha = 1.0 - iteration / num_iterations
                clip_range = 0.1 * alpha
                policy_loss_2 = b_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Clipped value loss
                value_loss1 = F.mse_loss(returns, value, reduction='none')
                value_loss2 = F.mse_loss(returns, torch.clamp(value, value - clip_range, value + clip_range), reduction='none')
                value_loss = torch.max(value_loss1, value_loss2).mean()

                # Compute total loss
                loss = policy_loss + ent_coef_c2 * -(m.entropy()).mean() + vf_coef_c1 * value_loss

                # Clip the gradient and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()
'''