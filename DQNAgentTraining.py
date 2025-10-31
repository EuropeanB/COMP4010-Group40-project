import gymnasium as gym
from Agents.DQNAgent import DQNAgent
import torch
import random
import numpy as np


def save_model(agent, name):
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'target_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'explore_rate': agent.explore_rate,
        'total_steps': agent.total_steps,
        'memory_size': len(agent.memory),
    }, f"Models/{name}.pth")


def load_model(agent, model_path):
    checkpoint = torch.load(model_path)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.explore_rate = checkpoint['explore_rate']
    agent.total_steps = checkpoint['total_steps']

    # Set to eval mode for consistent behavior
    agent.model.eval()
    agent.target_model.eval()


def build_summary(episode, score, cod, rewards, steps, win, explore_rate):
    output = str(episode) + " " * (3 - len(str(episode))) + " | "
    output += "Win! " if win else "Loss! "
    output += "Score: " + str(score) + " " * (3 - len(str(score))) + " | "
    output += "Cause of death: " + ("None" if win else cod) + " " * (20 - (4 if win else len(str(cod)))) + " | "
    output += "Total Reward: " + str(round(rewards, 2)) + " " * (8 - len(str(round(rewards, 2)))) + " | "
    output += "Total Steps: " + str(steps) + " " * (5 - len(str(steps))) + " | "
    output += "Explore Rate: " + str(explore_rate)
    return output


def test_model(episodes, model_path):
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode='human')
    unw_env = env.unwrapped

    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(board_dim, player_dim, action_size)
    load_model(agent, model_path)

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            legal_mask = unw_env.get_legal_moves_mask(state)
            action = agent.act(state, legal_mask, training=False)

            # Apply action
            next_state, _, terminated, truncated, info = env.step(action)

            # Print information
            print(info)

            # Update state
            state = next_state


def train_model(episodes):
    # Environment Setup
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode=None)
    unw_env = env.unwrapped

    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n

    # Create agent with frame stacking
    agent = DQNAgent(board_dim, player_dim, action_size)

    # Episode tracking stats
    total_reward = 0
    total_steps = 0
    total_episodes = 0
    sanity_check = np.float32(-1000.0)

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        # Start game
        while not (terminated or truncated):
            # Action selection and execution
            legal_mask = unw_env.get_legal_moves_mask(state)
            action = agent.act(state, legal_mask)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Sanity check
            if reward < sanity_check:
                print("WOAH OOPS! ILLEGAL MOVE FOR SOME REASON")

            # Store the experience
            agent.remember(state, action, reward, next_state, terminated)

            # Train on batch experiences
            agent.replay()

            # Update state
            state = next_state

            # Update tracking stats
            total_reward += reward
            total_steps += 1

        # Track training stats
        total_episodes += 1
        if (episode + 1) % 500 == 0:
            print(f"Episode: {episode} | Average Reward: {round(total_reward / total_steps, 2)} ({total_reward} reward over {total_steps} steps) | Average episode length: {round(total_steps / total_episodes, 2)}")
            total_steps = 0
            total_reward = 0
            total_episodes = 0

        if (episode + 1) % 5000 == 0:
            save_model(agent, f'{episode}_checkpoint')

    # Save model at the end
    save_model(agent, 'best_model')