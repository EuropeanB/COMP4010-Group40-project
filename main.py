import DQNAgentTraining
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

                # ---- TEMP TESTING ----
                '''print(f"New Legal Moves:")
                mask = obs['mask']
                for r in range(10):
                    line = ""
                    for c in range(13):
                        line += (str(int(mask[r * 13 + c]))) + " "
                    print(line)
                print(mask[-1])
                input("Hit enter")'''
                # ---- DONE TEMP TESTING ----

                with torch.no_grad():
                    logits, _ = actor_critic(board_obs, player_obs)
                    # --- TEMP TESTING MASKING ---
                    mask = torch.as_tensor(obs['mask'], dtype=torch.float32).to(device)
                    masked_logits = logits + (mask - 1) * 1e9
                    action = torch.argmax(masked_logits, dim=-1).item()
                    # --- DONE TEMP TESTING MASKING ---

                    #action = torch.argmax(logits, dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                print(info)
                print(f'REWARD: {reward}')
                #input("Hit enter")
