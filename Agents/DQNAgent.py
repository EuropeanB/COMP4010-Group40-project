import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super(DQN, self).__init__()

        self.channels, self.rows, self.cols = board_shape

        # CNN for board processing
        self.board_conv = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()
        )

        # Calculate CNN output size
        self.conv_out_size = 32 * self.rows * self.cols

        # Fully connected layers for board output + player processing
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + player_dim, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def _raw_forward(self, board_obs, player_obs):
        # board_obs: [batch, channels, rows, cols]
        # Process the board with CNN
        board_features = self.board_conv(board_obs)
        board_features = board_features.reshape(board_features.size(0), -1)

        # Concatenate with player features
        combined = torch.cat([board_features, player_obs], dim=1)

        # Process with the fully connected layer
        return self.fc(combined)

    @staticmethod
    def _bias_forward(board_obs, player_obs):
        SAFE_IDX = 5 # Should align with environment
        SAFE_BIAS = 5.0
        ONE_HP = (1 / 20) # Should align with environment
        HP_BIAS = 5.0

        B = board_obs.size(0)

        # Board bias
        safe_mask = board_obs[:, SAFE_IDX, :, :].reshape(B, -1)
        board_bias = safe_mask * SAFE_BIAS

        # Create output tensor
        player_bias = torch.zeros((B, 1), device=board_obs.device)
        q_bias = torch.cat([board_bias, player_bias], dim=1)

        return q_bias

    def forward(self, board_obs, player_obs):
        raw_vals = self._raw_forward(board_obs, player_obs)
        bias_vals = self._bias_forward(board_obs, player_obs)
        return raw_vals + bias_vals


class DQNAgent:
    def __init__(self, board_dim, player_dim, action_dim):
        # Initialization
        self.board_dim = board_dim
        self.player_dim = player_dim
        self.action_dim = action_dim
        self.total_steps = 0
        # Initialize agent memory
        self.memory = deque(maxlen=100_000)

        # --- AGENT HYPERPARAMETERS ---
        self.discount = 0 #0.93
        self.batch_size = 64
        self.learning_rate = 0.0001

        self.explore_rate = 1.0 # Start at 100%
        self.min_explore_rate = 0.15 # Keep at 10% exploration
        self.explore_decay_steps = 200_000  # Reach minimum exploration in 20k steps
        # self.explore_rate_decay = (self.min_explore_rate / self.explore_rate) ** (1 / self.explore_decay_steps)

        self.learning_starts = 5000 # Start learning after 1k steps
        self.target_update_freq = 2000  # Update target model every 1k steps
        self.train_frequency = 4 # Train on memories every 4 steps

        # Select device
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create and compile our models
        self.model = DQN(self.board_dim, self.player_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.board_dim, self.player_dim, self.action_dim).to(self.device)
        self._update_target_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

        # Preallocate inf tensor for legal masking
        self.neg_inf = -1e9


    def _update_target_model(self):
        """
        Every target_update_freq steps, the target model should copy the main model's weights

        :return: None
        """
        self.target_model.load_state_dict(self.model.state_dict())


    def _decay_exploration(self):
        """
        Decay the agent's exploration rate.

        :return: None
        """
        decay_amount = (self.explore_rate - self.min_explore_rate) / self.explore_decay_steps
        self.explore_rate = max(self.min_explore_rate, self.explore_rate - decay_amount)


    def remember(self, state, action, reward, next_state, terminal):
        """
        Adds observation to memory to be trained on later.

        :param state: State of the environment
        :param action: The action taken in the state
        :param reward: The reward given after the action
        :param next_state: The state that arose as a result of the action
        :param terminal: True if the next state is terminal, false otherwise
        :return: None
        """
        self.memory.append((state, action, reward, next_state, terminal))


    def act(self, state, training=True):
        """
        Action is chosen with tradeoff between exploration (taking random action)
        and exploitation (taking action maximizing Q), determined by explore_rate.

        :param state: The state on which to take action.
        :param training: True if the agent wishes to explore, False otherwise
        :return: The choice made.
        """
        # Decay exploration
        if training:
            self._decay_exploration()

        # Get the legal mask from the state
        legal_mask = state['mask']

        # Exploration (Choose random, legal action)
        if training and np.random.random() <= self.explore_rate:
            legal_actions = np.where(legal_mask)[0] # Get indices of legal actions
            return np.random.choice(legal_actions), None

        # Exploitation
        with torch.no_grad():
            board_tensor = torch.as_tensor(state['board'], dtype=torch.float32, device=self.device).unsqueeze(0)
            player_tensor = torch.as_tensor(state['player'], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            # Calculate Q-Values
            q_values = self.model(board_tensor, player_tensor)

            # Apply Masking
            masked_q = q_values.masked_fill(~mask_tensor, self.neg_inf)

            # Return best action
            return masked_q.argmax(dim=1).item(), masked_q


    def replay(self):
        """
        This function replays previous states and trains the model on them.
        The reason it's done like this is so that we can perform batch training
        on the model instead of a single instance at a time. Additionally, this
        removes any temporal dependencies that might arise as a result of training
        on single instances in order

        :return: The history when fit occurs, None otherwise
        """
        self.total_steps += 1

        # Check if we shouldn't be replaying
        if self.total_steps < self.learning_starts or len(self.memory) < self.batch_size:
            return self.explore_rate

        # Only train every train_frequency steps
        if self.total_steps % self.train_frequency != 0:
            return self.explore_rate

        # Combined Experience Replay
        batch = random.sample(self.memory, self.batch_size)

        board_states = []
        player_states = []
        board_next_states = []
        player_next_states = []
        mask_next_states = []
        actions = []
        rewards = []
        terminals = []

        for state, action, reward, next_state, terminal in batch:
            board_states.append(state['board'])
            player_states.append(state['player'])
            board_next_states.append(next_state['board'])
            player_next_states.append(next_state['player'])
            mask_next_states.append(next_state['mask'])
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)

        # Convert to torch datatypes
        board_states = torch.as_tensor(np.array(board_states), dtype=torch.float32, device=self.device)
        player_states = torch.as_tensor(np.array(player_states), dtype=torch.float32, device=self.device)
        board_next_states = torch.as_tensor(np.array(board_next_states), dtype=torch.float32, device=self.device)
        player_next_states = torch.as_tensor(np.array(player_next_states), dtype=torch.float32, device=self.device)
        mask_next_states = torch.as_tensor(np.array(mask_next_states), dtype=torch.bool, device=self.device)

        actions = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        terminals = torch.as_tensor(np.array(terminals), dtype=torch.bool, device=self.device).unsqueeze(1)

        # Compute Q-values for current states (Don't need to mask here, because we only take Q(s,a))
        q_values = self.model(board_states, player_states)
        q_values = q_values.gather(1, actions)

        # Compute target Q-values (using target model)
        with torch.no_grad():
            q_next_online = self.model(board_next_states, player_next_states) # Online network chooses actions
            q_next_online_masked = q_next_online.masked_fill(~mask_next_states, self.neg_inf) # Mask illegal actions
            next_actions = q_next_online_masked.argmax(dim=1, keepdim=True) # Choose next action

            q_next_target_all = self.target_model(board_next_states, player_next_states) # Target network evaluates those actions
            q_next_target_masked = q_next_target_all.masked_fill(~mask_next_states, self.neg_inf) # Mask illegal actions in target network
            max_next_q = q_next_target_masked.gather(1, next_actions)

            # Bellman target: if terminal, target = r , otherwise r + gamma * max_next_q
            targets = torch.where(terminals, rewards, rewards + self.discount * max_next_q)

        # Compute loss and optimize
        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

        self.optimizer.step()

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_model()

        return self.explore_rate
