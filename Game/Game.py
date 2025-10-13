from Actors import Actors
from DungeonGenerator import DungeonGenerator


class Game:
    def __init__(self):
        self.ROWS = 10
        self.COLS = 13

        # Lazy hard coding but if it works it works
        self.DRAGON_POSITION = (4, 6)  # Dragon always at (4,6)
        self.LEGAL_EGG_POSITIONS = [  # Dragon Egg always one away from Dragon
            (3, 5), (3, 6), (3, 7), (4, 5), (4, 7), (5, 5), (5, 6), (5, 7)
        ]
        self.EDGE_SPOTS = (
                [(0, col) for col in range(1, 12)] +
                [(row, 0) for row in range(1, 9)] +
                [(9, col) for col in range(1, 12)] +
                [(row, 12) for row in range(1, 9)]
        )

        self.dungeon_generator = DungeonGenerator(self.ROWS, self.COLS)
        self.board = None


    def reset_dungeon(self):
        """
        Resets the dungeon with a new board
        """
        self.board = False
        while not self.board:
            self.board = self.dungeon_generator.generate_dungeon()



    def __str__(self):
        output = "-" * 120 + '\n'
        for i in range(self.ROWS):
            for j in range(self.COLS):
                actor = self.board[i][j].actor
                if actor == Actors.EMPTY:
                    output += "EMPTY    | "
                elif actor == Actors.DRAGON:
                    output += "DRAGON   | "
                elif actor == Actors.DRAGON_EGG:
                    output += "EGG      | "
                elif actor == Actors.WIZARD:
                    output += "WIZARD   | "
                elif actor == Actors.BIG_SLIME:
                    output += "B. SLIME | "
                elif actor == Actors.MINE_KING:
                    output += "M. KING  | "
                elif actor == Actors.GIANT:
                    output += "GIANT    | "
                elif actor == Actors.CHEST:
                    output += "CHEST    | "
                elif actor == Actors.MINOTAUR:
                    output += "MINOTAUR | "
                elif actor == Actors.WALL:
                    output += "WALL     | "
                elif actor == Actors.GUARD:
                    output += "GUARDIAN | "
                elif actor == Actors.GARGOYLE:
                    output += "GARGOYLE | "
                elif actor == Actors.GAZER:
                    output += "GAZER    | "
                elif actor == Actors.MEDIKIT:
                    output += "MEDIKIT  | "
                elif actor == Actors.GNOME:
                    output += "GNOME    | "
                else:
                    output += "ERROR    | "
            output += '\n'
        return output

if __name__ == "__main__":
    game = Game()
    game.reset_dungeon()
    print(game)