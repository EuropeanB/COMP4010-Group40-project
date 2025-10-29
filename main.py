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

if __name__ == "__main__":
    episodes = 5000
    DQNAgentTraining.train_model(episodes)
    #DQNAgentTraining.test_model(episodes, "Models/best_model.pth")

    #REINFORCEAgentTraining.train_model(episodes)
    #REINFORCEAgentTraining.test_model(episodes, "Models/best_model.pth")