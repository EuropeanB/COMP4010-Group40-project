import gymnasium as gym
from Agents.DQNAgent import DQNAgent
import torch
import random


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
    output += "Total Reward: " + str(round(rewards, 2)) + " " * (5 - len(str(round(rewards, 2)))) + " | "
    output += "Total Steps: " + str(steps) + " " * (5 - len(str(steps))) + " | "
    output += "Explore Rate: " + str(explore_rate)
    return output


def test_model(episodes, model_path):
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode='human')
    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(board_dim, player_dim, action_size)
    load_model(agent, model_path)


    last_action = None
    count = 0

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = False

        while not terminated:
            action = agent.act(state, training=False)

            if last_action == action:
                count += 1
            else:
                count = 0
                last_action = action

            if count == 5:
                print("Choose random action, locked in")
                action = random.randrange(0, 131)
                last_action = action


            next_state, _, terminated, _, info = env.step(action)
            print(info)



            state = next_state


def train_model(episodes):
    # Environment Setup
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode=None)
    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(board_dim, player_dim, action_size)

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False

        # Episode tracking stats
        total_reward = 0
        total_steps = 0
        explore_rate = 0

        # Start game
        while not terminated:
            # Action selection and execution
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store the experience
            agent.remember(state, action, reward, next_state, terminated)

            # Train on batch experiences
            explore_rate = agent.replay()

            # Update state
            state = next_state

            # Update tracking stats
            total_reward += reward
            total_steps += 1

        # Track training stats
        won = info['hp'] > 0
        print(build_summary(episode, info['score'], info['last touched'], total_reward, total_steps, won, explore_rate))

    # Save model at the end
    save_model(agent, 'best_model')