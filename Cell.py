from enum import Enum
import numpy as np

class CellType(Enum):
    """
    Enum representing the different types of cells in DragonSweeper

    UNKNOWN: Unexplored cell
    EMPTY: Revealed safe cell with no items
    SAFE: Item that can be safely collected
    BRICK: Obstacle that costs HP to break
    HEALING_SCROLL: Health restoration item
    CHEST: Container that may contain items or mimics
    DRAGON: The final boss
    ENEMY: Any enemy other than the dragon
    HIDDEN: Under the effect of the gazer so value is hidden
    """
    UNKNOWN = 0
    EMPTY = 1
    SAFE = 2
    BRICK = 3
    HEALING_SCROLL = 4
    CHEST = 5
    DRAGON = 6
    ENEMY = 7
    HIDDEN = 8


class Cell:
    """
    Represents a single cell in the DragonSweeper game grid.

    Each cell tracks:
    - Whether it has been revealed
    - The sum of surrounding enemy power (or zero if unknown)
    - The number of surrounding bombs (or zero if unknown)
    - The type of content in the cell
    """

    def __init__(self, revealed=0, adj_power=0, cell_power=0, adj_bombs=0, status=CellType.UNKNOWN):
        """
        Initialize a Cell with its current state

        :param revealed: 1 if the cell is revealed, 0 otherwise
        :param adj_power: Sum of adjacent power, 0 if unknown or hidden
        :param cell_power: Power of the enemy in this cell, 0 if not an enemy
        :param adj_bombs: Total surrounding bombs, 0 if unknown
        :param status: CellType indicating what the cell contains
        """
        self.revealed = revealed
        self.adj_power = adj_power
        self.cell_power = cell_power
        self.adj_bombs = adj_bombs
        self.status = status

    def reset(self):
        """
        Reset the cell to its initial unknown state
        """
        self.revealed = 0
        self.adj_power = 0
        self.cell_power = 0
        self.adj_bombs = 0
        self.status = CellType.UNKNOWN

    def get_one_hot(self):
        """
        Get the one-hot encoding of the cell type for observation

        The encoding is an 9-element array where each position corresponds to a CellType:
        [UNKNOWN, EMPTY, SAFE, BRICK, HEALING_SCROLL, CHEST, DRAGON, ENEMY, HIDDEN]

        :return: numpy array of shape (9,) with dtype float32
        """
        one_hot = np.zeros(9, dtype=np.float32)
        one_hot[self.status.value] = 1

        return one_hot
