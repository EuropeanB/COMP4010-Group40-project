import gymnasium as gym
import torch
from Agents.PPOAgent import Environments
from Agents.PPOAgent import ActorCritic
from Agents.PPOAgent import PPO


def RunPPOAgent(training, save_directory=None, test_model=None, num_tests=20_000, test_render_mode='human'):
    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if training:
        num_actors = 8
        envs = Environments(num_actors)
        test_env = gym.make("Dragonsweeper-v0", render_mode=None)

        board_dim = envs.envs[0].observation_space['board'].shape
        player_dim = envs.envs[0].observation_space['player'].shape[0]
        action_size = envs.envs[0].action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)
        PPO(envs, test_env, actor_critic, save_directory, device=device)

    else:
        env = gym.make("Dragonsweeper-v0", render_mode=test_render_mode)
        board_dim = env.observation_space['board'].shape
        player_dim = env.observation_space['player'].shape[0]
        action_size = env.action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)
        state_dict = torch.load(test_model, weights_only=True)
        actor_critic.load_state_dict(state_dict)
        actor_critic.eval()

        for _ in range(num_tests):
            obs, info = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                board_obs = torch.as_tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
                player_obs = torch.as_tensor(obs['player'], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = actor_critic(board_obs, player_obs)
                    '''output = ""
                    for r in range(10):
                        for c in range(13):
                            output += f"{logits[0][r * 13 + c]:.2f} "
                        output += '\n'
                    print(output)
                    print(logits[0][-1])'''


                    mask = torch.as_tensor(obs['mask'], dtype=torch.float32).to(device)
                    masked_logits = logits + (mask - 1) * 1e9
                    action = torch.argmax(masked_logits, dim=-1).item()
                    m = torch.distributions.Categorical(logits=masked_logits)

                obs, reward, terminated, truncated, info = env.step(action)
                print(info)
                print(f'REWARD: {reward}')
                #input("continue")