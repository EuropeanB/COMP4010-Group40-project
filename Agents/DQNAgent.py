import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0


class DQN(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super(DQN, self).__init__()

        rows, cols, channels = board_shape

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

        # Fully connected layers for board output + player processing
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + player_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )


    def forward(self, board_obs, player_obs):
        # board_obs: [batch, rows, cols, channels] -> [batch, channels, rows, cols]
        board_obs = board_obs.contiguous().permute(0, 3, 1, 2)

        # Process the board with CNN
        board_features = self.board_conv(board_obs)
        board_features = board_features.reshape(board_features.size(0), -1)

        # Concatenate with player features
        combined = torch.cat([board_features, player_obs], dim=1)

        # Process with the fully connected layer
        return self.fc(combined)


class DQNAgent:
    def __init__(self, board_dim, player_dim, action_dim):
        # Initialization
        self.board_dim = board_dim
        self.player_dim = player_dim
        self.action_dim = action_dim
        self.total_steps = 0
        self.latest_experience = None

        # Initialize agent memory
        self.memory = deque(maxlen=10000)

        # --- AGENT HYPERPARAMETERS ---
        self.discount = 0.99
        self.batch_size = 64
        self.learning_rate = 0.001

        self.explore_rate = 1.0 # Start at 100%
        self.min_explore_rate = 0.1 # Keep at 10% exploration
        self.explore_decay_steps = 100_000 # Reach minimum exploration in 20k steps
        self.explore_rate_decay = (self.min_explore_rate / self.explore_rate) ** (1 / self.explore_decay_steps)

        self.learning_starts = 1000 # Start learning after 1k steps
        self.target_update_freq = 1000  # Update target model every 1k steps
        self.train_frequency = 1 # Train on memories every 8 steps


        # Create and compile our models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.board_dim, self.player_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.board_dim, self.player_dim, self.action_dim).to(self.device)
        self._update_target_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

        # Preallocate inf tensor for legal masking
        self._inf_tensor = torch.full((1,), -float('inf'), device=self.device)


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
        self.explore_rate = max(self.min_explore_rate, self.explore_rate * self.explore_rate_decay)


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
        self.latest_experience = (state, action, reward, next_state, terminal)


    def act(self, state, legal_mask, training=True):
        """
        Action is chosen with tradeoff between exploration (taking random action)
        and exploitation (taking action maximizing Q), determined by explore_rate.

        :param state: The state on which to take action.
        :param training: True if the agent wishes to explore, False otherwise
        :return: The choice made.
        """
        # Exploration
        if training and np.random.random() <= self.explore_rate:
            legal_actions = np.where(legal_mask)[0] # Get indices of legal actions
            return np.random.choice(legal_actions)

        # Exploitation
        with torch.no_grad():
            board_tensor = torch.as_tensor(state['board'], device=self.device).unsqueeze(0)
            player_tensor = torch.as_tensor(state['player'], device=self.device).unsqueeze(0)
            q_values = self.model(board_tensor, player_tensor)

            # Convert legal mask to tensor and apply masking
            legal_mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            masked_q = torch.where(legal_mask_tensor, q_values, self._inf_tensor)

            return masked_q.argmax().item()


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
        batch = random.sample(self.memory, self.batch_size - 1)
        batch.append(self.latest_experience)

        board_states = []
        player_states = []
        board_next_states = []
        player_next_states = []
        actions = []
        rewards = []
        terminals = []

        for state, action, reward, next_state, terminal in batch:
            board_states.append(state['board'])
            player_states.append(state['player'])
            board_next_states.append(next_state['board'])
            player_next_states.append(next_state['player'])
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)

        # Convert to torch datatypes
        board_states = torch.as_tensor(np.array(board_states), dtype=torch.float32, device=self.device)
        player_states = torch.as_tensor(np.array(player_states), dtype=torch.float32, device=self.device)
        board_next_states = torch.as_tensor(np.array(board_next_states), dtype=torch.float32, device=self.device)
        player_next_states = torch.as_tensor(np.array(player_next_states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        terminals = torch.as_tensor(np.array(terminals), dtype=torch.bool, device=self.device).unsqueeze(1)

        # Compute Q-values for current states
        q_values = self.model(board_states, player_states)
        q_values = q_values.gather(1, actions)

        # Compute target Q-values (using target model)
        with torch.no_grad():
            next_actions = self.model(board_next_states, player_next_states).argmax(1, keepdim=True)
            max_next_q = self.target_model(board_next_states, player_next_states).gather(1, next_actions)
            targets = torch.where(terminals, rewards, rewards + self.discount * max_next_q)

        # Compute loss and optimize
        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_model()

        # Decay exploration
        self._decay_exploration()

        return self.explore_rate
