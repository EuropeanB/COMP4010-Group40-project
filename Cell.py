from Actors import Actors

class Cell:
    """
    Cell representation for the game (not the agent)
    """
    def __init__(self, actor=Actors.EMPTY):
        self.actor = actor
        self.adj_power = 0
        self.revealed = False
        self.power = 0
        self.xp = 0
        self.contains_medikit = False # Only relevant if cell is a chest
        self.obscured = False

        self.previously_rat_king = False # Required because rat king goes to xp first then goes to rat scroll
        self.previously_mine_king = False # Required because mine king goes to xp first then goes to reveal mine scroll
        self.previously_wizard = False # Required because wizard goes to xp first then goes to reveal slime scroll
        self.previously_giant = False # Required because giant goes to xp first then goes to medikit
        self.previously_dragon = False # Required because dragon goes to xp first then goes to crown
        self.previous_power = 0 # Required because cell values only update once the xp is collected, not when the enemy is killed

    def set_adj_power(self, power):
        self.adj_power = power
