import gymnasium as gym
from Agents.REINFORCEAgent import REINFORCEAgent
import torch
import random


def save_model(agent, name):
    torch.save({
        'policy_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, f"Models/{name}.pth")


def load_model(agent, model_path):
    """Load the policy network"""
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.policy_net.eval()


def build_summary(episode, score, cod, rewards, steps, win):
    output = str(episode) + " " * (3 - len(str(episode))) + " | "
    output += "Win! " if win else "Loss! "
    output += "Score: " + str(score) + " " * (3 - len(str(score))) + " | "
    output += "Cause of death: " + ("None" if win else cod) + " " * (20 - (4 if win else len(str(cod)))) + " | "
    output += "Total Reward: " + str(round(rewards, 2)) + " " * (5 - len(str(round(rewards, 2)))) + " | "
    output += "Total Steps: " + str(steps) + " " * (5 - len(str(steps))) + " | "
    return output


def test_model(episodes, model_path):
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode='human')
    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(board_dim, player_dim, action_size)
    load_model(agent, model_path)

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        last_action = None
        count = 0

        print(f"=== Test Episode {episode + 1} ===")

        while not (terminated or truncated):
            action = agent.act(state, training=False)  # Use greedy policy

            # Verify it isn't soft locked using the same broken action
            if last_action == action:
                count += 1
            else:
                count = 0
                last_action = action
            if count == 7:
                print("WARNING, AGENT TOOK SAME ACTION 7 TIMES IN A ROW. CHOOSING RANDOM ACTION")
                action = random.randrange(0, 131)
                last_action = action


            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            # Optional: Add delay to watch the game
            import time
            time.sleep(0.2)

        print(f"Test Episode {episode + 1}: {steps} steps, total reward: {total_reward:.1f}")
        if info.get('hp', 0) > 0:
            print("*** VICTORY! ***")
        else:
            print(f"Defeated by: {info.get('last touched', 'Unknown')}")


def train_model(episodes):
    # Environment Setup
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    env = gym.make("Dragonsweeper-v0", render_mode=None)
    board_dim = env.observation_space['board'].shape
    player_dim = env.observation_space['player'].shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(board_dim, player_dim, action_size)

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        total_reward = 0
        steps = 0

        # Run episode
        while not (terminated or truncated):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store reward for this step
            agent.remember(reward)

            state = next_state
            total_reward += reward
            steps += 1

        # Update policy at the end of episode
        loss = agent.update()

        # Track training stats
        won = info['hp'] > 0
        print(build_summary(episode, info['score'], info['last touched'], total_reward, steps, won))


    # Save model at the end
    save_model(agent, 'best_model')