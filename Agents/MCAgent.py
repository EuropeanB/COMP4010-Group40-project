# MCAgent.py

# Run the program with: C:/Python311/python.exe Agents/MCAgent.py --use-seed --seed 123

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
import sys
from torch import optim
import argparse
import matplotlib.pyplot as plt
import time
import random
_sysrand = random.SystemRandom()

# Add parent directory to Python path to import register_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Environment import DragonSweeperEnv
from Game import Game

# --- Hyperparameters (defaults; can be overridden via CLI) ---
GAMMA = 0.99
EPS_RANDOM = 0.1
DEFAULT_LR = 1e-4
NUM_EPISODES = 20000
SAVE_EVERY = 500
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
USE_BASELINE = True  # Use critic head as baseline to reduce variance

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
        # Input: cell features (6) + player state (2) = 8
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
        cells = board_obs.permute(0, 2, 3, 1).reshape(batch * num_cells, channels)  # [B * 130, 6]

        # Broadcast player state to each cell
        player_expanded = player_obs.unsqueeze(1).expand(batch, num_cells, self.player_dim)
        player_expanded = player_expanded.reshape(batch * num_cells, self.player_dim)  # [B * 130, 2]

        # Concatenate so each cell sees its features and player state
        cell_input = torch.cat([cells, player_expanded], dim=1)  # [B * 130, 8]

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


# --- Setup ---

def make_env():
    return DragonSweeperEnv()

def epsilon_random():
    return _sysrand.random()

def set_global_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def moving_average(data, window = 50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def compute_returns(rewards, gamma=GAMMA, device=DEVICE):
    """Compute discounted returns (Monte Carlo, full-episode)."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)

def safe_normalize(tensor, eps=1e-8):
    """Normalize a 1D tensor but avoid division by zero / NaNs."""
    if tensor.numel() == 0:
        return tensor
    std = tensor.std()
    if torch.isfinite(std) and std.item() >= eps:
        return (tensor - tensor.mean()) / (std + eps)
    else:
        return tensor - tensor.mean()

def visualize_policy_matplotlib(model, env, episode_num, device):
    # Reset environment deterministically based on episode number
    obs, _ = env.reset(seed=episode_num)
    board = torch.as_tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
    player = torch.as_tensor(obs['player'], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(obs['mask'], dtype=torch.float32, device=device).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        logits, value = model(board, player)

    # Mask invalid actions
    legal_logits = logits[0].clone()
    legal_logits[mask[0] == 0] = float("-inf")

    # Convert to probabilities
    probs = torch.softmax(legal_logits, dim=0).cpu().numpy()
    grid_probs = probs[:-1].reshape(env.ROWS, env.COLS)

    # Prepare danger map (for color shading)
    power_danger = obs["board"][env.POWER_DANGER_IDX]
    mine_danger = obs["board"][env.MINE_DANGER_IDX]
    wall_mask = obs["board"][env.WALL_IDX]

    # Create figure
    plt.figure(figsize=(12, 7))
    plt.title(f"Policy Heatmap at Episode {episode_num}\nValue Estimate = {value.item():.3f}")

    # Base heatmap = probability of clicking
    plt.imshow(grid_probs, cmap="Blues", vmin=0, vmax=grid_probs.max() + 1e-6)

    # Overlay danger:  
    # - walls = black  
    # - mine danger = red  
    # - high power danger = orange  
    danger_overlay = np.zeros((*grid_probs.shape, 3))  # RGB

    # Walls → black
    danger_overlay[wall_mask == 1] = (0, 0, 0)

    # Mine danger → red (blend based on danger level)
    if mine_danger.max() > 0:
        danger_overlay[..., 0] += mine_danger  # Red channel

    # Power danger → orange
    danger_overlay[..., 0] += power_danger * 0.6
    danger_overlay[..., 1] += power_danger * 0.3

    # Normalize to [0,1]
    danger_overlay = np.clip(danger_overlay, 0, 1)

    # Overlay danger shading with alpha
    plt.imshow(danger_overlay, alpha=0.4)

    # Put probability numbers on top
    for r in range(env.ROWS):
        for c in range(env.COLS):
            p = grid_probs[r, c]
            plt.text(c, r, f"{p:.2f}", va='center', ha='center',
                     color="white" if p < 0.2 else "black", fontsize=7)

    plt.colorbar(label="Policy Probability")
    plt.tight_layout()
    plt.show()


def train(args):

    # ---------------------------
    # SEEDING MODE
    # ---------------------------

    if args.use_seed:
        print(f"Seeding enabled with seed {args.seed}")
        set_global_seed(args.seed)
    else:
        print("Seeding disabled; using random seeds")

    env = make_env()
    if args.use_seed:
        env.reset(seed=args.seed)
    else:
        env.reset()

    board_shape = env.observation_space['board'].shape    # (rows, cols, channels)
    player_dim = env.observation_space['player'].shape[0] # e.g. 4
    action_size = env.action_space.n                     # e.g. 131
    print(f"Obs board shape: {board_shape}, player dim: {player_dim}, actions: {action_size}")

    model = ActorCritic(board_shape, player_dim, action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)

    running_avg_reward = None
    reward_history = []

    episode_times = []
    start_time = time.time()

    for ep in range(1, NUM_EPISODES + 1):

        

        episode_start_time = time.time()

        if args.use_seed:
            obs, info = env.reset(seed=args.seed)
        else:
            obs, info = env.reset()

        episode_logprobs = []
        episode_values = []
        episode_rewards = []
        episode_actions = []
        episode_entropies = []


        # play one full episode
        episode_length = 0
        while True:
            # board_np = np.transpose(obs['board'], (1, 2, 0))  # [channels, rows, cols]
            board_np = obs['board']  # already in [channels, rows, cols] format
            board = torch.as_tensor(board_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            player = torch.as_tensor(obs['player'], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, player_dim]

            # DEBUG: catch NaNs in observations
            if torch.isnan(board).any() or torch.isnan(player).any():
                print("NaN detected in observations! Dumping obs and exiting for debug:")
                print(obs)
                raise RuntimeError("NaN in observation tensors")

            logits, value = model(board, player)  # logits: [1, action_size], value: [1,1]

            # Legal action masking via environment's function (this is the authoritative source of legality)
            legal_mask_np = obs['mask']  # returns numpy bool array length action_size
            legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool, device=DEVICE)

            # If env reports no legal moves (shouldn't normally happen) -> fallback
            if not legal_mask.any():
                # fallback: use unmasked logits (still prefer to avoid crash)
                masked_logits = logits.clone()
            else:
                masked_logits = logits.clone()
                # Set illegal actions to a very large negative value (logit space)
                # ensure mask length matches logits
                assert masked_logits.shape[1] == legal_mask.shape[0], "legal_mask length mismatch with logits"
                masked_logits[0, ~legal_mask] = -1e10

            # Guard against NaNs / infs in masked_logits before constructing distribution
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                masked_logits = masked_logits.nan_to_num(nan=-1e10, posinf=1e10, neginf=-1e10)
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    # last-resort fallback
                    masked_logits = torch.zeros_like(masked_logits)

            # Epsilon-random exploration fallback
            eps = max(0.02, 0.2 * (1 - episode_length / 50))
            # if epsilon_random() < EPS_RANDOM:
            if epsilon_random() < eps:
                # print("Epsilon-random action selected!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # choose uniformly from legal actions
                if legal_mask.any():
                    legal_indices = np.flatnonzero(legal_mask_np)
                    chosen = int(np.random.choice(legal_indices))
                else:
                    # if no legal mask, fallback to argmax of logits
                    chosen = int(torch.argmax(masked_logits, dim=1).item())
                action = torch.tensor([chosen], device=DEVICE)
                # create small dummy logprob/entropy; we'll set them to something consistent
                logprob = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
                entropy = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
            else:
                # Sample action from masked logits distribution
                try:
                    m = torch.distributions.Categorical(logits=masked_logits)
                    action = m.sample()                 # shape [1]
                    logprob = m.log_prob(action)        # shape [1]
                    entropy = m.entropy()               # shape [1]
                except Exception as e:
                    # If sampling fails for some reason, fallback to greedy selection among legal actions
                    print(f"Sampling failed ({e}); falling back to greedy selection among legal actions.")
                    if legal_mask.any():
                        legal_logits = masked_logits.clone()
                        legal_logits[0, ~legal_mask] = -1e20
                        action_idx = int(torch.argmax(legal_logits, dim=1).item())
                    else:
                        action_idx = int(torch.argmax(logits, dim=1).item())
                    action = torch.tensor([action_idx], device=DEVICE)
                    logprob = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
                    entropy = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)

            # step
            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            episode_length += 1

            # store scalars/tensors (make them 1D tensors for stacking)
            episode_logprobs.append(logprob.squeeze(0))  # shape []
            episode_values.append(value.squeeze(-1).squeeze(0))       # value -> scalar tensor
            episode_rewards.append(float(reward))
            episode_actions.append(int(action.item()))
            episode_entropies.append(entropy.squeeze(0))

            obs = next_obs
            done = terminated or truncated
            if done:
                break

        # Episode finished -> compute returns and losses
        if len(episode_rewards) == 0:
            # nothing happened this episode (shouldn't normally occur) -> skip updates
            continue
            
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)

        returns = compute_returns(episode_rewards, gamma=args.gamma, device=DEVICE)  # [T]
        logprobs = torch.stack(episode_logprobs).to(DEVICE)      # [T]
        values = torch.stack(episode_values).to(DEVICE)          # [T]
        entropies = torch.stack(episode_entropies).to(DEVICE)    # [T] (may be zeros if eps-random used)

        # Policy loss (REINFORCE). If using baseline, subtract value estimates.
        if USE_BASELINE:
            advantages = returns - values.detach()
        else:
            advantages = returns.clone()

        # Normalize advantages (not returns)
        # adv_mean = advantages.mean()
        # adv_std = advantages.std()
        # if adv_std <= 1e-8 or not torch.isfinite(adv_std):
        #     advantages = advantages - adv_mean
        # else:
        #     advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = returns - values.detach()

        if ep % 200 == 0 and episode_length < 5:
            print("POWER_DANGER:")
            board_danger = obs['board'][0]  # channel 1
            print(board_danger)
            print("MINE_DANGER:")
            board_mine = obs['board'][1]  # channel 2
            print(board_mine)

        # if ep % 50 == 0:
        #     print(f"[REWARD DEBUG] ep {ep} mean return {returns.mean().item():.3f} ep_reward {ep_reward:.3f}")


        if ep % 50 == 0:
            print("Episode rewards:", episode_rewards)

        policy_loss = -(logprobs * advantages).mean()

        # Critic loss: MSE between returns and value estimates (Monte Carlo target)
        if USE_BASELINE:
            value_loss = F.mse_loss(values, returns)
        else:
            value_loss = torch.tensor(0.0, device=DEVICE)

        # Entropy term
        ent_coef = args.ent_coef
        if entropies.numel() > 0:
            entropy_term = entropies.mean()
        else:
            entropy_term = torch.tensor(0.0, device=DEVICE)

        # Full loss
        loss = policy_loss + args.vf_coef * value_loss - ent_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        ep_reward = sum(episode_rewards)
        reward_history.append(ep_reward)
        running_avg_reward = ep_reward if running_avg_reward is None else 0.99 * running_avg_reward + 0.01 * ep_reward

        # Logging
        if ep % args.log_every == 0:
            avg_recent = np.mean(reward_history[-100:]) if len(reward_history) >= 1 else 0

            # Compute timing stats
            avg_ep_time = np.mean(episode_times[-50:]) if len(episode_times) >= 1 else 0
            elapsed = time.time() - start_time
            remaining_episodes = NUM_EPISODES - ep
            eta_seconds = remaining_episodes * avg_ep_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60

            # print(
            #     f"Episode {ep}/{NUM_EPISODES} | "
            #     f"Reward: {ep_reward:.3f} | "
            #     f"Avg100: {avg_recent:.3f} | "
            #     f"Episode Length: {episode_length} | "
            #     f"Avg Time/Ep: {avg_ep_time:.2f}s | "
            #     f"Est. Time Left: {eta_minutes:.1f} min ({eta_hours:.2f} hr)"
            # )
        
        # if ep % 500 == 0 and ep > 0:
        #     visualize_policy_matplotlib(model, env, ep, DEVICE)

        # Save model periodically
        if ep % SAVE_EVERY == 0:
            save_path = os.path.join(log_dir, f"mc_agent_ep{ep}.pth")
            torch.save({
                "episode": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward_history": reward_history
            }, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    # Plot reward history


    plt.figure(figsize=(10,5))
    plt.plot(reward_history, alpha=0.3, label="Raw episode reward")
    plt.plot(moving_average(reward_history, 50), linewidth=2, label="Moving Average (50)")
    plt.title("Episode Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="Models_mc", help="dir to save models")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="weight for value loss (baseline)")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="grad clip norm")
    parser.add_argument("--log-every", type=int, default=10, help="print every N episodes")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.03, help="entropy coefficient")
    parser.add_argument("--eps-random", type=float, default=0.05, help="epsilon random exploration probability")
    parser.add_argument("--use-seed", action="store_true", help="Enable deterministic seeding for debugging")
    parser.add_argument("--seed", type=int, default=123, help="Seed used when --use-seed is enabled")
    args = parser.parse_args()

    # expose args values used inside train() by attribute names expected
    args.lr = args.lr
    args.vf_coef = args.vf_coef
    args.grad_clip = args.grad_clip
    args.log_every = args.log_every
    args.save_dir = args.save_dir
    args.gamma = args.gamma
    args.ent_coef = args.ent_coef
    args.eps_random = args.eps_random

    print("Device:", DEVICE)
    train(args)
