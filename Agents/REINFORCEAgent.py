import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, board_shape, player_dim, num_actions):
        super(PolicyNetwork, self).__init__()

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
        total_features = self.conv_out_size + player_dim

        # Policy network
        self.linear1 = nn.Linear(total_features, 512)
        self.linear2 = nn.Linear(512, num_actions)


    def forward(self, board_obs, player_obs):
        # board_obs: [batch, rows, cols, channels] -> [batch, channels, rows, cols]
        board_obs = board_obs.permute(0, 3, 1, 2)

        # Process the board with CNN
        board_features = self.board_conv(board_obs)
        board_features = board_features.reshape(board_features.size(0), -1)

        # Concatenate with player features
        combined = torch.cat([board_features, player_obs], dim=1)

        # Forward pass
        x = F.relu(self.linear1(combined))
        x = F.softmax(self.linear2(x), dim=1)
        return x


    def get_action(self, board_obs, player_obs):
        probs = self.forward(board_obs, player_obs)
        probs_np = probs.squeeze(0).detach().cpu().numpy()

        action = np.random.choice(len(probs_np), p=probs_np)
        log_prob = torch.log(probs.squeeze(0)[action])

        return action, log_prob


class REINFORCEAgent:
    def __init__(self, board_dim, player_dim, action_dim):
        self.board_dim = board_dim
        self.player_dim = player_dim
        self.action_dim = action_dim

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001

        # Network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(board_dim, player_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Episode storage
        self.saved_log_probs = []
        self.rewards = []


    def act(self, state, training=True):
        """
        Choose an action using the policy network
        """
        board_tensor = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
        player_tensor = torch.FloatTensor(state['player']).unsqueeze(0).to(self.device)

        if training:
            action, log_prob = self.policy_net.get_action(board_tensor, player_tensor)
            self.saved_log_probs.append(log_prob)
            return action
        else:
            # For testing: choose action with the highest probability
            with torch.no_grad():
                probs = self.policy_net(board_tensor, player_tensor)
                return probs.argmax().item()


    def remember(self, reward):
        """
        Store reward for the current step
        """
        self.rewards.append(reward)


    def update(self):
        """
        Update policy using REINFORCE at the end of episode
        """
        if not self.rewards:
            return None

        # Calculate returns
        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)

        # Only normalize if we have multiple different rewards
        if len(discounted_rewards) > 1 and discounted_rewards.std() > 1e-8:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # Otherwise, leave as-is (single step episode or all rewards same)


        # Calculate policy gradient
        policy_gradient = []
        for log_prob, Gt in zip(self.saved_log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        # Update
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

        # Clear
        self.saved_log_probs = []
        self.rewards = []

        return policy_gradient.item()