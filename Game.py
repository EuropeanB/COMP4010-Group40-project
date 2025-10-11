from enum import Enum
import random

# Same name as the other one but not the same thing
class Actors(Enum):
    NONE = 0
    EMPTY = 1
    ORB = 2
    SPELL_MAKE_ORB = 3
    MINE = 4
    MINE_KING = 5
    DRAGON = 6
    WALL = 7
    MIMIC = 8
    MEDIKIT = 9
    RAT_KING = 10
    RAT = 11
    SLIME = 12
    GARGOYLE = 13
    MINOTAUR = 14
    CHEST = 15
    SKELETON = 16
    TREASURE = 17
    SNAKE = 18
    GIANT = 19
    WIZARD = 20
    GAZER = 21
    SPELL_DISARM = 22
    BIG_SLIME = 23
    SPELL_REVEAL_RATS = 24
    SPELL_REVEAL_SLIMES = 25
    GNOME = 26
    BAT = 27
    GUARD = 28
    CROWN = 29
    FIDEL = 30
    DRAGON_EGG = 31


class Cell:
    def __init__(self, actor=Actors.EMPTY):
        self.actor = actor
        self.adj_power = 0
        self.revealed = 0

    def set_adj_power(self, power):
        self.adj_power = power


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

        self.board = [[Cell() for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.generateDungeon()

    def generateDungeon(self):
        # Remaining Positions
        rem = [(row, col) for row in range(10) for col in range(13)]

        # Place Dragon and Egg
        self.board[self.DRAGON_POSITION[0]][self.DRAGON_POSITION[1]].actor = Actors.DRAGON
        rem.remove((self.DRAGON_POSITION[0], self.DRAGON_POSITION[1]))
        egg_row, egg_col = random.choice(self.LEGAL_EGG_POSITIONS)
        self.board[egg_row][egg_col].actor = Actors.DRAGON_EGG
        rem.remove((egg_row, egg_col))

        # Place Wizard (Edge) and Big Slimes (Surrounding)
        wizard_spot = random.choice(self.EDGE_SPOTS)
        wizard_row, wizard_col = wizard_spot
        self.board[wizard_row][wizard_col].actor = Actors.WIZARD
        rem.remove((wizard_row, wizard_col))
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if 0 <= wizard_row + i <= 9 and 0 <= wizard_col + j <= 12:
                    if i == j == 0:
                        continue
                    self.board[wizard_row+i][wizard_col+j].actor = Actors.BIG_SLIME
                    rem.remove((wizard_row+i, wizard_col+j))

        # Place Mine King (Need to cover cases where blobs get placed in corners)
        mine_king_options = [(0, 0), (0, 12), (9, 0), (9, 12)]
        if wizard_spot in [(0, 1), (1, 0)]: # Near Top Left Corner
            mine_king_options.remove((0, 0))
        elif wizard_spot in [(8, 0), (9, 1)]: # Near Bottom Left Corner
            mine_king_options.remove((9, 0))
        elif wizard_spot in [(0, 11), (1, 12)]: # Near Top Right Corner
            mine_king_options.remove((0, 12))
        elif wizard_spot in [(8, 12), (9, 11)]: # Near Bottom Right Corner
            mine_king_options.remove((9, 12))

        mine_king_row, mine_king_col = random.choice(mine_king_options)
        self.board[mine_king_row][mine_king_col].actor = Actors.MINE_KING
        rem.remove((mine_king_row, mine_king_col))

        # Starting "Layer 3"

        # Place Giants (Romeo and Juliette) Must be same row, same distance from center
        while True:
            giant_row1, giant_col1 = random.choice([(row, col) for row, col in rem if col < 6])
            giant_col2 = 12 - giant_col1
            if (giant_row1, giant_col2) in rem:
                self.board[giant_row1][giant_col1].actor = Actors.GIANT
                self.board[giant_row1][giant_col2].actor = Actors.GIANT
                rem.remove((giant_row1, giant_col1))
                rem.remove((giant_row1, giant_col2))
                break


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
                else:
                    output += "ERROR    | "
            output += '\n'
        return output

if __name__ == "__main__":
    game = Game()
    print(game)