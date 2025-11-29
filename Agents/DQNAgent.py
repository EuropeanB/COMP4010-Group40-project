import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super(DQN, self).__init__()

        # Stored for convenience
        self.board_shape = board_shape
        self.player_dim = player_dim
        self.num_actions = num_actions

        channels, rows, cols = board_shape
        self.num_cells = rows * cols  # 130

        # ---- Per-cell MLP encoder (same as PPO) ----
        self.cell_encoder = nn.Sequential(
            nn.Linear(channels + player_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # ---- Board aggregator (same as PPO) ----
        self.board_aggregator = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # ---- Cell-based Q-head (per-tile Q-value) ----
        self.cell_q_head = nn.Linear(32, 1)

        # ---- Level-up Q-head ----
        self.level_q_head = nn.Sequential(
            nn.Linear(player_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, board_obs, player_obs):
        B, channels, rows, cols = board_obs.shape
        num_cells = rows * cols

        # Flatten board cells
        # board_obs → [B, R, C, channels] → [B*num_cells, channels]
        cell_features = (
            board_obs.permute(0, 2, 3, 1)
            .reshape(B * num_cells, channels)
        )

        # Repeat player state for every cell
        player_expanded = (
            player_obs.unsqueeze(1)
            .expand(B, num_cells, self.player_dim)
            .reshape(B * num_cells, self.player_dim)
        )

        # Per-cell input: concat board cell + player state
        cell_input = torch.cat([cell_features, player_expanded], dim=1)

        # Encode cells (same as PPO)
        cell_emb = self.cell_encoder(cell_input)  # [B*num_cells, 32]
        cell_emb = cell_emb.reshape(B, num_cells, 32)  # [B, 130, 32]

        # ---- Per-cell Q-values ----
        cell_q = self.cell_q_head(cell_emb).squeeze(-1)  # [B, 130]

        # ---- Board summary ----
        board_summary = self.board_aggregator(cell_emb.mean(dim=1))  # [B, 32]

        # ---- Level-up Q-value ----
        lvl_input = torch.cat([player_obs, board_summary], dim=1)  # [B, 34]
        lvl_q = self.level_q_head(lvl_input).squeeze(-1)  # [B]

        # ---- Combine ----
        q_values = torch.cat([cell_q, lvl_q.unsqueeze(1)], dim=1)  # [B, 131]

        return q_values


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
        self.discount = 0.99
        self.batch_size = 64
        self.learning_rate = 0.0001

        self.explore_rate = 1.0 # Start at 100%
        self.min_explore_rate = 0.05 # Keep at 10% exploration
        self.explore_decay_steps = 50_000  # Reach minimum exploration in 20k steps
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
