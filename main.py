from Environment import DragonSweeperEnv
from Agents.DummyAgent import DummyAgent

if __name__ == "__main__":
    num_games = 20
    render_mode = "human"
    #render_mode = None

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
            print(info)

        print("GAME ENDED")

    env.close()
