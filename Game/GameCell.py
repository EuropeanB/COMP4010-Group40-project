from Actors import Actors

class Cell:
    """
    Cell representation for the game (not the agent)
    """
    def __init__(self, actor=Actors.EMPTY):
        self.actor = actor
        self.adj_power = 0
        self.revealed = 0

    def set_adj_power(self, power):
        self.adj_power = power