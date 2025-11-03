'''from Environment import DragonSweeperEnv
from Agents.DummyAgent import DummyAgent

if __name__ == "__main__":
    num_games = 200
    render_mode = "human"

    env = DragonSweeperEnv(render_mode=render_mode)
    agent = DummyAgent(env.action_space)

    for _ in range(num_games):
        print("STARTING GAME")

        obs, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            print(action, reward)

            if reward > 0:
                input("Hit enter")

        print("GAME ENDED")

    env.close()'''

import DQNAgentTraining
import REINFORCEAgentTraining
from Agents.PPOAgent import Environments
from Agents.PPOAgent import ActorCritic
from Agents.PPOAgent import PPO
import gymnasium as gym
import torch

if __name__ == "__main__":
    train = False # True for training, false for testing

    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if train:
        num_actors = 8
        envs = Environments(num_actors)

        board_dim = envs.envs[0].observation_space['board'].shape
        player_dim = envs.envs[0].observation_space['player'].shape[0]
        action_size = envs.envs[0].action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)
        PPO(envs, actor_critic, device=device)

    else:
        env = gym.make("Dragonsweeper-v0", render_mode='human')
        best_actor_critic = torch.load("best_agent")
        best_actor_critic.to(device)
        best_actor_critic.eval()

        num_episodes = 10
        for _ in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                board_obs = torch.as_tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
                player_obs = torch.as_tensor(obs['player'], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = best_actor_critic(board_obs, player_obs)
                    m = torch.distributions.Categorical(logits=logits)
                    action = m.sample().item()

                obs, reward, terminated, truncated, info = env.step(action)
                print(f'REWARD: {reward}')
