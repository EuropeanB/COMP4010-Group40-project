# MCAgent.py (batched Monte-Carlo version)
# Run: C:/Python311/python.exe Agents/MCAgent_batch.py --num-envs 8 --batch-episodes 8 --use-seed --seed 123

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

def visualize_policy_matplotlib(model, env, episode_num, device):
    # Reset environment deterministically based on episode number
    obs, _ = env.reset(seed=episode_num)
    board = torch.as_tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
    player = torch.as_tensor(obs['player'], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(obs['mask'], dtype=torch.float32, device=device).unsqueeze(0)

    # Forward pass
    logits, value = model(board, player)

    # Mask invalid actions
    legal_logits = logits[0].clone()
    legal_logits[mask[0] == 0] = float("-inf")

    # Convert to probabilities
    probs = torch.softmax(legal_logits, dim=0).detach().cpu().numpy()
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

    # Create multiple envs (manual parallel)
    num_envs = args.num_envs
    batch_episodes = args.batch_episodes
    assert batch_episodes >= 1 and num_envs >= 1, "num-envs and batch-episodes must be >= 1"

    envs = [make_env() for _ in range(num_envs)]

    # Optionally seed each env deterministically
    if args.use_seed:
        for i, e in enumerate(envs):
            e.reset(seed=args.seed + i)

    # Single env used only for visualization when needed
    viz_env = make_env()

    # Get shapes from one env
    sample_obs, _ = envs[0].reset()
    board_shape = envs[0].observation_space['board'].shape    # (channels, rows, cols)
    player_dim = envs[0].observation_space['player'].shape[0] # e.g. 2
    action_size = envs[0].action_space.n                     # e.g. 131
    print(f"Obs board shape: {board_shape}, player dim: {player_dim}, actions: {action_size}")

    model = ActorCritic(board_shape, player_dim, action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)

    reward_history = []
    running_avg_reward = None
    episode_times = []
    start_time = time.time()

    # Per-env current observation / done flags / per-episode buffers
    curr_obs = [None] * num_envs
    done_flags = [False] * num_envs

    # Initialize obs
    for i, e in enumerate(envs):
        if args.use_seed:
            obs, info = e.reset(seed=args.seed + i)
        else:
            obs, info = e.reset()
        curr_obs[i] = obs
        done_flags[i] = False

    global_episode = 0

    ep = 0
    while ep < NUM_EPISODES:
        batch_start_time = time.time()

        # Buffers to accumulate across the batch of completed episodes
        batch_logprobs = []
        batch_values = []
        batch_returns = []
        batch_entropies = []
        batch_actions = []
        batch_episode_rewards = []  # scalar per episode for logging
        batch_episode_lengths = []

        finished_episodes = 0

        # Per-env trajectory buffers (accumulate until that env finishes)
        traj_logprobs = [[] for _ in range(num_envs)]
        traj_values = [[] for _ in range(num_envs)]
        traj_rewards = [[] for _ in range(num_envs)]
        traj_entropies = [[] for _ in range(num_envs)]
        traj_actions = [[] for _ in range(num_envs)]

        # Run until we collect 'batch_episodes' completed episodes
        while finished_episodes < batch_episodes:
            for i in range(num_envs):
                if finished_episodes >= batch_episodes:
                    break

                if done_flags[i]:
                    continue  # env already finished this cycle; will be reset once we use it

                obs = curr_obs[i]
                # Prepare tensors
                board_np = obs['board']
                board = torch.as_tensor(board_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                player = torch.as_tensor(obs['player'], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Forward
                logits, value = model(board, player)

                # Legal action masking
                legal_mask_np = obs['mask']
                legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool, device=DEVICE)

                if not legal_mask.any():
                    masked_logits = logits.clone()
                else:
                    masked_logits = logits.clone()
                    masked_logits[0, ~legal_mask] = -1e10

                # fix NaN/inf in logits
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    masked_logits = masked_logits.nan_to_num(nan=-1e10, posinf=1e10, neginf=-1e10)
                    if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                        masked_logits = torch.zeros_like(masked_logits)

                # Epsilon-random exploration
                eps = max(0.02, 0.2 * (1 - len(traj_rewards[i]) / 50))
                if epsilon_random() < eps:
                    if legal_mask.any():
                        legal_indices = np.flatnonzero(legal_mask_np)
                        chosen = int(np.random.choice(legal_indices))
                    else:
                        chosen = int(torch.argmax(masked_logits, dim=1).item())
                    action = torch.tensor([chosen], device=DEVICE)
                    logprob = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
                    entropy = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
                else:
                    try:
                        m = torch.distributions.Categorical(logits=masked_logits)
                        action = m.sample()
                        logprob = m.log_prob(action)
                        entropy = m.entropy()
                    except Exception as e:
                        if legal_mask.any():
                            legal_logits = masked_logits.clone()
                            legal_logits[0, ~legal_mask] = -1e20
                            action_idx = int(torch.argmax(legal_logits, dim=1).item())
                        else:
                            action_idx = int(torch.argmax(logits, dim=1).item())
                        action = torch.tensor([action_idx], device=DEVICE)
                        logprob = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
                        entropy = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)

                # Step env
                next_obs, reward, terminated, truncated, info = envs[i].step(int(action.item()))

                # Save step data into per-env trajectory buffers
                traj_logprobs[i].append(logprob.squeeze(0))
                traj_values[i].append(value.squeeze(-1).squeeze(0))
                traj_rewards[i].append(float(reward))
                traj_entropies[i].append(entropy.squeeze(0))
                traj_actions[i].append(int(action.item()))

                curr_obs[i] = next_obs
                done = terminated or truncated

                if done:
                    # Episode finished for env i -> compute returns and append to batch lists
                    episode_rewards = traj_rewards[i]
                    if len(episode_rewards) == 0:
                        # skip empty episodes (shouldn't happen)
                        done_flags[i] = True
                        finished_episodes += 1
                        # reset the env immediately for next usage
                        if args.use_seed:
                            curr_obs[i], _ = envs[i].reset(seed=args.seed + i + finished_episodes)
                        else:
                            curr_obs[i], _ = envs[i].reset()
                        done_flags[i] = False
                        traj_logprobs[i] = []
                        traj_values[i] = []
                        traj_rewards[i] = []
                        traj_entropies[i] = []
                        traj_actions[i] = []
                        continue

                    returns = compute_returns(episode_rewards, gamma=args.gamma, device=DEVICE)  # tensor [T]
                    logprobs = torch.stack(traj_logprobs[i]).to(DEVICE)
                    values = torch.stack(traj_values[i]).to(DEVICE)
                    entropies = torch.stack(traj_entropies[i]).to(DEVICE)

                    # Save to batch buffers (concatenate across episodes)
                    batch_logprobs.append(logprobs)
                    batch_values.append(values)
                    batch_returns.append(returns)
                    batch_entropies.append(entropies)
                    batch_actions.append(traj_actions[i])
                    batch_episode_rewards.append(sum(episode_rewards))
                    batch_episode_lengths.append(len(episode_rewards))

                    finished_episodes += 1
                    global_episode += 1
                    ep += 1

                    # reset that env immediately so it can contribute another episode in this batch if necessary
                    if args.use_seed:
                        curr_obs[i], _ = envs[i].reset(seed=args.seed + i + global_episode)
                    else:
                        curr_obs[i], _ = envs[i].reset()

                    # clear its trajectory buffers
                    traj_logprobs[i] = []
                    traj_values[i] = []
                    traj_rewards[i] = []
                    traj_entropies[i] = []
                    traj_actions[i] = []

                    # done_flags stays False because we reset immediately

                # guard: if we've reached desired total episodes, break the outer loop
                if finished_episodes >= batch_episodes:
                    break

        # Now we have collected `batch_episodes` full episodes. Build loss and update once.

        # Concatenate episode tensors into one long timeline for gradient computation
        if len(batch_logprobs) == 0:
            # nothing to learn from (shouldn't happen) -> continue
            continue

        # Flatten all episodes into single tensors
        all_logprobs = torch.cat(batch_logprobs, dim=0)       # [Sum_T]
        all_values = torch.cat(batch_values, dim=0)           # [Sum_T]
        all_returns = torch.cat(batch_returns, dim=0)         # [Sum_T]
        all_entropies = torch.cat(batch_entropies, dim=0)     # [Sum_T]

        # Advantages (MC baseline if requested)
        if USE_BASELINE:
            advantages = all_returns - all_values.detach()
        else:
            advantages = all_returns.clone()

        policy_loss = -(all_logprobs * advantages).mean()

        if USE_BASELINE:
            value_loss = F.mse_loss(all_values, all_returns)
        else:
            value_loss = torch.tensor(0.0, device=DEVICE)

        ent_coef = args.ent_coef
        if all_entropies.numel() > 0:
            entropy_term = all_entropies.mean()
        else:
            entropy_term = torch.tensor(0.0, device=DEVICE)

        loss = policy_loss + args.vf_coef * value_loss - ent_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging & bookkeeping after batch update
        for r in batch_episode_rewards:
            reward_history.append(r)
            running_avg_reward = r if running_avg_reward is None else 0.99 * running_avg_reward + 0.01 * r

        avg_recent = np.mean(reward_history[-100:]) if len(reward_history) >= 1 else 0.0
        avg_batch_reward = np.mean(batch_episode_rewards)

        batch_time = time.time() - batch_start_time
        episode_times.append(batch_time / max(1, len(batch_episode_rewards)))

        if global_episode % args.log_every == 0 or global_episode <= 10:
            print(f"[Batch Update] GlobalEp {global_episode} | Episodes in batch {len(batch_episode_rewards)} | "
                  f"Avg batch reward {avg_batch_reward:.3f} | Avg100 {avg_recent:.3f} | Loss {loss.item():.4f}")

        # Save checkpoints periodically
        if global_episode % SAVE_EVERY == 0:
            save_path = os.path.join(log_dir, f"mc_agent_ep{global_episode}.pth")
            torch.save({
                "episode": global_episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward_history": reward_history
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

        # Optionally visualize occasionally with single env
        # if global_episode % 500 == 0:
        #     try:
        #         visualize_policy_matplotlib(model, viz_env, global_episode, DEVICE)
        #     except Exception as e:
        #         print("Visualization failed:", e)

        # Stop condition
        if ep >= NUM_EPISODES:
            break

    # finished training loop
    plt.figure(figsize=(10,5))
    plt.plot(reward_history, alpha=0.3, label="Raw episode reward")
    if len(reward_history) >= 50:
        plt.plot(moving_average(reward_history, 50), linewidth=2, label="Moving Average (50)")
    plt.title("Episode Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    for e in envs:
        e.close()
    viz_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="Models_mc", help="dir to save models")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="weight for value loss (baseline)")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="grad clip norm")
    parser.add_argument("--log-every", type=int, default=10, help="print every N episodes")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.03, help="entropy coefficient")
    parser.add_argument("--num-envs", type=int, default=8, help="number of parallel envs to run")
    parser.add_argument("--batch-episodes", type=int, default=8, help="number of completed episodes to collect per update")
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
    args.num_envs = args.num_envs
    args.batch_episodes = args.batch_episodes

    print("Device:", DEVICE)
    train(args)
