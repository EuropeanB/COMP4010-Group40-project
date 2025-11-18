import DQNAgentTraining
from Agents.PPOAgent import Environments
from Agents.PPOAgent import ActorCritic
from Agents.PPOAgent import PPO
from Agents.DQNAgent import DQNAgent
import gymnasium as gym
import torch
import numpy as np

from collections import deque

'''def print_board(board):
    output = ""
    for i in range(11):
        output += f"CHANNEL: {i}\n"
        for r in range(10):
            for c in range(13):
                output += f"{board[i, r, c]} "
            output += '\n'
        output += '\n\n'
    print(output)

def play(env, agent, episodes):
    episode_rewards = deque(maxlen=episodes)
    episode_lengths = deque(maxlen=episodes)
    illegal_moves = 0
    orb_first = 0
    firsts = 0

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False

        #print_board(state['board'])
        #input("Continue")

        # ---- Metric tracking ----
        steps = 0
        total_rewards = 0
        first_move = True
        # -------------------------

        while not (terminated or truncated):
            action, masked_q_values = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            print(reward)
            input("Continue")

            # ---- Print Q-Values for Testing (First move only for now) ----

            # ------------------------------------

            # ---- Metric tracking ----
            if reward == -2.0:
                illegal_moves += 1
            if first_move:
                first_move = False
                firsts += 1
                if info['last touched'] == "ORB":
                    orb_first += 1
            steps += 1
            total_rewards += reward
            # -------------------------

        # ---- Metric tracking ----
        episode_rewards.append(total_rewards)
        episode_lengths.append(steps)
        # -------------------------

    print(f"Testing! {episodes} episodes"
          f" | AVG. Total Episode Reward: {np.mean(list(episode_rewards))}"
          f" | AVG. Episode Length: {np.mean(list(episode_lengths))}"
          f" | Illegal Moves: {illegal_moves} (Truncating!)"
          f" | Orb First Rate: {orb_first / firsts} ({orb_first} / {firsts})")


def save_model(model, name):
    torch.save(model.state_dict(), f"Models/{name}.pth")

def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    #render_mode = 'human'
    render_mode = None
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode=render_mode)
    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(board_dim, player_dim, action_size)

    #load_model(agent.model, f"Models/best_agent.pth")
    #play(env, agent, 1000)
    #exit(0)

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
        if episode % 5000 == 0 and episode > 0:
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

        # Play the model every N episodes
        if episode % 5000 == 0 and episode > 0:
            test_env = gym.make("Dragonsweeper-v0", render_mode=None)
            play(test_env, agent, 100)'''


import Environment
if __name__ == "__main__":
    train = True # True for training, false for testing

    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    #device = torch.device("cpu") # For now
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if train:
        num_actors = 8
        envs = Environments(num_actors)

        board_dim = envs.envs[0].observation_space['board'].shape
        player_dim = envs.envs[0].observation_space['player'].shape[0]
        action_size = envs.envs[0].action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)
        PPO(envs, actor_critic, "Models", device=device)

    else:
        env = gym.make("Dragonsweeper-v0", render_mode='human')
        board_dim = env.observation_space['board'].shape
        player_dim = env.observation_space['player'].shape[0]
        action_size = env.action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)
        state_dict = torch.load("Models/best_agent.pth", weights_only=True)
        actor_critic.load_state_dict(state_dict)
        actor_critic.eval()

        num_episodes = 100
        for _ in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                board_obs = torch.as_tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
                player_obs = torch.as_tensor(obs['player'], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = actor_critic(board_obs, player_obs)
                    mask = torch.as_tensor(obs['mask'], dtype=torch.float32).to(device)
                    masked_logits = logits + (mask - 1) * 1e9
                    action = torch.argmax(masked_logits, dim=-1).item()
                    m = torch.distributions.Categorical(logits=masked_logits)

                obs, reward, terminated, truncated, info = env.step(action)
                print(info)
                print(f'REWARD: {reward}')
