import gymnasium as gym
import torch

from Agents.PPOAgent import Environments   # shared function from PPO
from Agents.PPOAgent import ActorCritic    # shared function from PPO
from Agents.ACAgent import AC

def RunACAgent(training, save_directory="Models", test_model=None, num_tests=20000, test_render_mode='human'):

    gym.register(id='Dragonsweeper-v0', entry_point='Environment:DragonSweeperEnv')
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print(f"[AC] Using device: {device}")

    # Training
    if training:

        num_actors = 8
        envs = Environments(num_actors)

        board_dim = envs.envs[0].observation_space["board"].shape
        player_dim = envs.envs[0].observation_space["player"].shape[0]
        action_size = envs.envs[0].action_space.n

        print(f"[AC] board={board_dim}, player={player_dim}, actions={action_size}")

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)

        AC(
            envs=envs,
            actor_critic=actor_critic,
            save_path=save_directory,
            device=device
        )

    # Testing
    else:

        env = gym.make("Dragonsweeper-v0", render_mode=test_render_mode)

        board_dim = env.observation_space["board"].shape
        player_dim = env.observation_space["player"].shape[0]
        action_size = env.action_space.n

        actor_critic = ActorCritic(board_dim, player_dim, action_size).to(device)

        print(f"[AC] Loading model: {test_model}")
        state_dict = torch.load(test_model, weights_only=True)
        actor_critic.load_state_dict(state_dict)
        actor_critic.eval()

        for _ in range(num_tests):

            obs, info = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):

                board_obs = torch.as_tensor(
                    obs["board"], dtype=torch.float32, device=device
                ).unsqueeze(0)

                player_obs = torch.as_tensor(
                    obs["player"], dtype=torch.float32, device=device
                ).unsqueeze(0)

                mask = torch.as_tensor(
                    obs["mask"], dtype=torch.bool, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = actor_critic(board_obs, player_obs)
                    masked_logits = logits.masked_fill(~mask, -1e9)
                    action = torch.argmax(masked_logits, dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)

                print(info)
                print(f"REWARD: {reward}")

        print("[AC] Testing completed")
