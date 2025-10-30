import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions, frame_stack=4):
        super(DQN, self).__init__()

        rows, cols, channels = board_shape
        self.frame_stack = frame_stack
        self.input_channels = channels * frame_stack

        # CNN for board processing
        self.board_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
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
        board_obs = board_obs.permute(0, 3, 1, 2)

        # Process the board with CNN
        board_features = self.board_conv(board_obs)
        board_features = board_features.reshape(board_features.size(0), -1)

        # Concatenate with player features
        combined = torch.cat([board_features, player_obs], dim=1)

        # Process with the fully connected layer
        return self.fc(combined)



class DQNAgent:
    def __init__(self, board_dim, player_dim, action_dim, frame_stack=4):
        # Initialization
        self.board_dim = board_dim
        self.player_dim = player_dim
        self.action_dim = action_dim
        self.frame_stack = frame_stack
        self.total_steps = 0
        self.latest_experience = None

        # Initialize frame stack buffer
        self.frame_buffer = deque(maxlen=frame_stack)

        # Initialize agent memory
        self.memory = deque(maxlen=10000)

        # --- AGENT HYPERPARAMETERS ---
        self.discount = 0.99
        self.batch_size = 64
        self.learning_rate = 0.005

        self.explore_rate = 1.0 # Start at 100%
        self.min_explore_rate = 0.2 # Keep at 10% exploration
        self.explore_decay_steps = 20000 # Reach minimum exploration in 20k steps
        self.explore_rate_decay = (self.min_explore_rate / self.explore_rate) ** (1 / self.explore_decay_steps)

        self.learning_starts = 1000 # Start learning after 1k steps
        self.target_update_freq = 1000  # Update target model every 1k steps
        self.train_frequency = 1 # Train on memories every step


        # Create and compile our models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.board_dim, self.player_dim, self.action_dim, self.frame_stack).to(self.device)
        self.target_model = DQN(self.board_dim, self.player_dim, self.action_dim, self.frame_stack).to(self.device)
        self._update_target_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)


    def _get_stacked_state(self, state):
        """
        Stack current state with previous frames
        """
        # Add current frame to buffer
        self.frame_buffer.append(state['board'])

        # If we don't have enough frames yet, repeat the first frame
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft(state['board'])

        # Stack along channel dimension (axis=2)
        stacked_boards = np.concatenate(list(self.frame_buffer), axis=2)

        return {
            'board': stacked_boards,  # Shape: (rows, cols, channels * frame_stack)
            'player': state['player']  # Use current player stats
        }


    def reset_episode(self):
        """
        Reset frame buffer at start of new episode

        :return: None
        """
        self.frame_buffer.clear()


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
        stacked_state = self._get_stacked_state(state)
        stacked_next_state = self._get_stacked_state(next_state)

        self.memory.append((stacked_state, action, reward, stacked_next_state, terminal))
        self.latest_experience = (stacked_state, action, reward, stacked_next_state, terminal)


    def act(self, state, legal_mask, training=True):
        """
        Action is chosen with tradeoff between exploration (taking random action)
        and exploitation (taking action maximizing Q), determined by explore_rate.

        :param state: The state on which to take action.
        :param training: True if the agent wishes to explore, False otherwise
        :return: The choice made.
        """
        stacked_state = self._get_stacked_state(state)

        # Exploration
        if training and np.random.random() <= self.explore_rate:
            legal_actions = np.where(legal_mask)[0] # Get indices of legal actions
            return np.random.choice(legal_actions)

        # Exploitation
        with torch.no_grad():
            board_tensor = torch.FloatTensor(stacked_state['board']).unsqueeze(0).to(self.device)
            player_tensor = torch.FloatTensor(stacked_state['player']).unsqueeze(0).to(self.device)
            q_values = self.model(board_tensor, player_tensor)

            # Convert legal mask to tensor and apply masking
            legal_mask_tensor = torch.BoolTensor(legal_mask).to(self.device)
            masked_q = torch.where(legal_mask_tensor, q_values, torch.tensor(-float('inf')).to(self.device))

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
        board_states = torch.FloatTensor(np.array(board_states)).to(self.device)
        player_states = torch.FloatTensor(np.array(player_states)).to(self.device)
        board_next_states = torch.FloatTensor(np.array(board_next_states)).to(self.device)
        player_next_states = torch.FloatTensor(np.array(player_next_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        terminals = torch.BoolTensor(np.array(terminals)).unsqueeze(1).to(self.device)

        # Compute Q-values for current states
        q_values = self.model(board_states, player_states).gather(1, actions)

        # Compute target Q-values (using target model)
        with torch.no_grad():
            '''next_q = self.target_model(board_next_states, player_next_states)
            max_next_q = next_q.max(1)[0].unsqueeze(1)
            targets = torch.where(terminals, rewards, rewards + self.discount * max_next_q)'''
            next_actions = self.model(board_next_states, player_next_states).argmax(1, keepdim=True)
            max_next_q = self.target_model(board_next_states, player_next_states).gather(1, next_actions)
            targets = torch.where(terminals, rewards, rewards + self.discount * max_next_q)

        # Compute loss and optimize
        loss = nn.MSELoss()(q_values, targets)
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
