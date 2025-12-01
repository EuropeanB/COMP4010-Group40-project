import gymnasium as gym
import torch
from collections import deque
from Agents.DQNAgent import DQNAgent
import numpy as np


def save_model(model, name):
    torch.save(model.state_dict(), f"Models/{name}.pth")


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def RunDQNAgent(training, save_directory=None, test_model=None, num_tests=20_000, test_render_mode='human'):
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if training:
        env = gym.make("Dragonsweeper-v0", render_mode=None)

        board_dim = env.observation_space['board'].shape
        player_dim = env.observation_space['player'].shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(board_dim, player_dim, action_size)

        episodes = 10_000_000

        # ---- Metric Tracking ----
        episode_reward_list = deque(maxlen=2000)
        step_reward_list = deque(maxlen=5000)
        episode_length_list = deque(maxlen=2000)
        first_move_orb_list = deque(maxlen=2000)
        max_episode_reward = -100
        explore_rate = 0
        # -------------------------

        for episode in range(episodes):
            state, _ = env.reset()
            terminated = truncated = False

            # ---- Metric Tracking ----
            steps = 0
            episode_rewards = 0
            first_move = True
            # -------------------------

            # Start game
            while not (terminated or truncated):
                # Action selection
                action, _ = agent.act(state, training=True)

                # Action execution
                next_state, reward, terminated, truncated, info = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, (terminated or truncated))

                # Train on batch experience
                explore_rate = agent.replay()

                # State transition
                state = next_state

                # ---- Step Over - Metric Tracking ----
                steps += 1
                step_reward_list.append(reward)
                episode_rewards += reward
                if first_move:
                    first_move = False
                    first_move_orb_list.append(1 if info['last touched'] == 'ORB' else 0)
                # -------------------------------------


            # ---- Episode over - Metric Tracking ----
            episode_reward_list.append(episode_rewards)
            episode_length_list.append(steps)
            if episode_rewards > max_episode_reward:
                max_episode_reward = episode_rewards
                save_model(agent.model, "best_agent")
            # ----------------------------------------


            # Log metrics every N episodes
            if episode % 500 == 0 and episode > 0:
                avg_ep_reward = np.mean(list(episode_reward_list))
                avg_ep_length = np.mean(list(episode_length_list))
                avg_step_reward = np.mean(list(step_reward_list))
                orb_rate = (sum(first_move_orb_list) / len(first_move_orb_list))

                print(
                    f"\nIteration {episode}"
                    f" | Avg episode reward: {avg_ep_reward:.3f}"
                    f" | Avg step reward: {avg_step_reward:.3f}"
                    f" | Avg ep length: {avg_ep_length:.1f}"
                    f" | ORB first-move rate: {orb_rate * 100:.1f}%"
                    f" | Explore rate: {explore_rate:.3f}"
                )

            # Save model every N episodes
            if episode % 50_000 == 0 and episode > 0:
                save_model(agent.model, f"{episode}")
