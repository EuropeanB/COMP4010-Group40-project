from Environment import DragonSweeperEnv
from GameVisual import GameVisual

env = DragonSweeperEnv()
env.reset()
#step = env.step(1)
vis = GameVisual()

vis.set_game(env.game)