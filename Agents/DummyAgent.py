import numpy as np

class DummyAgent:
    """
    A simple dummy agent for DragonSweeper.
    Chooses random actions from the environment's action space.
    """
    def __init__(self, action_space):
        """
        :param action_space: Gymnasium action space from environment
        """
        self.action_space = action_space

    def select_action(self, observation):
        """
        Returns an action based on the observation.
        Currently random, can be replaced by real agent logic later.

        :param observation: Environment observation dict
        :return: action (int)
        """
        temp = observation
        return self.action_space.sample()
