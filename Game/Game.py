from Actors import Actors
from DungeonGenerator import DungeonGenerator
import random
import math


class Game:
    def __init__(self):
        self.ROWS = 10
        self.COLS = 13
        self.ORB_REVEAL = [(-2, 0),(-1, -1),(-1, 0),(-1, 1),(0, -2),(0, -1),(0, 1),(0, 2),(1, -1),(1, 0),(1, 1),(2, 0)]

        self.XP_REQUIREMENTS = [4, 5, 7, 9, 9, 10, 12, 12, 12, 15, 18, 21, 21, 25]

        self.score = 0
        self.level = 1
        self.xp = 0
        self.max_health = 5
        self.curr_health = 5

        self.dungeon_generator = DungeonGenerator(self.ROWS, self.COLS)
        self.board = None


    def reset_game(self):
        """
        Resets the dungeon with a new board and reset player stats
        """
        self.score = 0
        self.level = 1
        self.xp = 0
        self.max_health = 5
        self.curr_health = 5

        self.board = False
        while not self.board:
            self.board = self.dungeon_generator.generate_dungeon()


    def level_up(self):
        """
        Levels up the character if they have enough XP. Gain a heart container every 2 levels, to a max of 19.

        :return: True if level up, False if unable
        """
        curr_level = self.level - 1 if self.level < len(self.XP_REQUIREMENTS) else len(self.XP_REQUIREMENTS) - 1

        # Check if player can even level up
        if self.xp < self.XP_REQUIREMENTS[curr_level]:
            return False

        # Remove required xp
        self.xp -= self.XP_REQUIREMENTS[curr_level]

        # Add a heart if even level is even and if below 19 total hearts
        if self.level % 2 == 0 and self.max_health < 19:
            self.max_health += 1

        # Level up and heal
        self.level += 1
        self.curr_health = self.max_health

        return True


    def touch_square(self, row, col):
        """
        Handles all logic related to clicking on a square.

        :param row: The row of the cell to click
        :param col: The col of the cell to click
        :return: (Alive, Win) tuple. Win is True if crown is grabbed
        """
        cell = self.board[row][col]

        if cell.actor in [
            Actors.RAT, Actors.BAT, Actors.SKELETON, Actors.GARGOYLE, Actors.SLIME,
            Actors.MINOTAUR, Actors.GUARD, Actors.BIG_SLIME
        ]:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.actor = Actors.XP
            cell.xp = cell.power
            cell.power = 0
            cell.revealed = True

        elif cell.actor == Actors.MIMIC: # It's important to pass this to the agent as a chest, as they won't know
            if cell.revealed is False:
                cell.revealed = True
            else:
                self._damage_player(cell.power)
                cell.previous_power = cell.power
                cell.actor = Actors.XP
                cell.xp = cell.power
                cell.power = 0

        elif cell.actor == Actors.EMPTY or cell.actor == Actors.NONE:
            if not cell.revealed:
                cell.revealed = True
            else:
                return True, False # Perhaps do something here to indicate it essentially chose a pointless move?

        elif cell.actor == Actors.MINE:
            if cell.power == 0: # Disabled (Clicking a disabled bomb sets it to XP even if hidden)
                cell.actor = Actors.XP
                cell.xp = 3
                cell.revealed = True
            else:
                self._damage_player(cell.power)
                cell.previous_power = cell.power
                cell.actor = Actors.EMPTY
                cell.power = 0
                cell.revealed = True

        elif cell.actor == Actors.XP:
            self.xp += cell.xp
            self.score += cell.xp
            cell.xp = 0
            cell.revealed = True

            if cell.previous_power > 0:
                self._update_surrounding_cells(row, col, cell.previous_power)
                cell.previous_power = 0

            if cell.previously_rat_king:
                cell.actor = Actors.SPELL_REVEAL_RATS
                cell.previously_rat_king = False
            elif cell.previously_mine_king:
                cell.actor = Actors.SPELL_DISARM
                cell.previously_mine_king = False
            elif cell.previously_wizard:
                cell.actor = Actors.SPELL_REVEAL_SLIMES
                cell.previously_wizard = False
            elif cell.previously_giant:
                cell.actor = Actors.MEDIKIT
                cell.previously_giant = False
            elif cell.previously_dragon:
                cell.actor = Actors.CROWN
                cell.previously_dragon = False
            else:
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.DRAGON:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.actor = Actors.XP
            cell.power = 0
            cell.previously_dragon = True
            cell.revealed = True

        elif cell.actor == Actors.DRAGON_EGG:
            cell.actor = Actors.XP
            cell.xp = 3
            cell.revealed = True

        elif cell.actor == Actors.CROWN:
            cell.actor = Actors.EMPTY
            return True, True

        elif cell.actor == Actors.CHEST:
            if cell.revealed is False:
                cell.revealed = True
            elif cell.contains_medikit:
                cell.contains_medikit = False
                cell.actor = Actors.MEDIKIT
            else:
                cell.actor = Actors.XP
                cell.xp = 5

        elif cell.actor == Actors.MEDIKIT:
            if cell.revealed is False:
                cell.revealed = True
            else:
                self.curr_health = self.max_health
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.MINE_KING:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.actor = Actors.XP
            cell.power = 0
            cell.previously_mine_king = True
            cell.revealed = True

        elif cell.actor == Actors.SPELL_DISARM:
            # Sets all revealed bombs to XP, all hidden bombs become zero power (update board)
            if cell.revealed is False:
                cell.revealed = True
            else:
                self._disarm_mines()
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.RAT_KING:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.actor = Actors.XP
            cell.power = 0
            cell.previously_rat_king = True
            cell.revealed = True

        elif cell.actor == Actors.SPELL_REVEAL_RATS:
            if cell.revealed is False:
                cell.revealed = True
            else:
                self._reveal_rats()
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.WIZARD:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.actor = Actors.XP
            cell.power = 0
            cell.previously_wizard = True
            cell.revealed = True

        elif cell.actor == Actors.SPELL_REVEAL_SLIMES:
            if cell.revealed is False:
                cell.revealed = True
            else:
                self._reveal_slimes()
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.ORB:
            self._orb_reveal(row, col)
            cell.actor = Actors.EMPTY
            cell.revealed = True

        elif cell.actor == Actors.WALL:
            if not cell.revealed:
                cell.revealed = True
            else:
                self._damage_player(1)
                cell.power -= 1
                if cell.power == 0: # Transform if broken
                    cell.actor = Actors.XP
                    cell.previous_power = 0
                    cell.xp = 1
                    cell.revealed = True

        elif cell.actor == Actors.GIANT:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.actor = Actors.XP
            cell.power = 0
            cell.previously_giant = True
            cell.revealed = True

        elif cell.actor == Actors.SPELL_MAKE_ORB:
            if cell.revealed is False:
                cell.revealed = True
            else:
                self._cast_orb_scroll()
                cell.actor = Actors.EMPTY

        elif cell.actor == Actors.GAZER:
            self._damage_player(cell.power)
            cell.previous_power = cell.power
            cell.xp = cell.power
            cell.power = 0
            cell.actor = Actors.XP
            cell.revealed = True

            # Un-obscure surrounding areas
            for (add_row, add_col) in self.ORB_REVEAL:
                new_row = row + add_row
                new_col = col + add_col

                if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                    continue

                self.board[new_row][new_col].obscured = False

            # Re-obscure surrounding area for the remaining gazer (if there's overlap in the gazers)
            for i in range(self.ROWS):
                for j in range(self.COLS):
                    if self.board[i][j].actor == Actors.GAZER:
                        for (add_row, add_col) in self.ORB_REVEAL:
                            new_row = i + add_row
                            new_col = j + add_col

                            if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                                continue

                            self.board[new_row][new_col].obscured = True
                        break

        elif cell.actor == Actors.GNOME:
            # Reveal the current square and move the gnome
            cell.actor = Actors.EMPTY
            cell.revealed = True

            # Find all remaining medikits
            medikits = []
            for i in range(self.ROWS):
                for j in range(self.COLS):
                    if self.board[i][j].actor == Actors.MEDIKIT:
                        medikits.append((i, j))

            # Find all empty, unrevealed squares
            candidates = []
            for i in range(self.ROWS):
                for j in range(self.COLS):
                    cell = self.board[i][j]
                    if not cell.revealed and cell.actor == Actors.EMPTY:
                        candidates.append((i, j))
            random.shuffle(candidates)

            # If no candidate squares, then gnome is turned to XP
            if len(candidates) == 0:
                cell.actor = Actors.XP
                cell.xp = 9
                return True, False

            # Find closest to a medikit
            scores = []
            for candidate in candidates:
                score = 10000
                for medikit in medikits:
                    temp_score = math.sqrt((candidate[1] - medikit[1]) ** 2 + (candidate[0] - medikit[0]) ** 2)
                    score = temp_score if temp_score < score else score
                scores.append(score)

            # The closest to a medikit becomes the gnome
            min_score = min(scores)
            min_index = scores.index(min_score)
            self.board[candidates[min_index][0]][candidates[min_index][1]].actor = Actors.GNOME

        else:
            print("ERROR! UNKNOWN ACTOR SELECTED")


        return self.curr_health >= 0, False


    def _damage_player(self, monster_power):
        """
        Damages the player with the monster power up to -1 HP.

        :param monster_power: The amount of damage to deal to the player
        """
        self.curr_health = max(-1, self.curr_health - monster_power)


    def _disarm_mines(self):
        """
        Disarms all mines by setting their power to zero. Turn revealed bombs to XP directly,
        and recalculate surrounding values.
        """
        for i in range(self.ROWS):
            for j in range(self.COLS):
                cell = self.board[i][j]
                if cell.actor == Actors.MINE:
                    # Adjust power for neighbouring cells
                    for (adj_row, adj_col) in self.get_surrounding_cells((i, j), True):
                        self.board[adj_row][adj_col].adj_power -= 100

                    # Turn to XP if already revealed
                    if cell.revealed is True:
                        cell.actor = Actors.XP
                        cell.xp = 3

                    # Set power to zero (defused)
                    cell.power = 0


    def _reveal_rats(self):
        """
        Sets status to revealed for all rats
        """
        for row in self.board:
            for cell in row:
                if cell.actor == Actors.RAT:
                    cell.revealed = True


    def _reveal_slimes(self):
        """
        Sets status to revealed for all slimes
        """
        for row in self.board:
            for cell in row:
                if cell.actor == Actors.SLIME or cell.actor == Actors.BIG_SLIME:
                    cell.revealed = True


    def _orb_reveal(self, row, col):
        """
        Reveals all squares surrounding the orb (within 2 Euclidean Distance)

        :param row: The row of the orb
        :param col: The col of the orb
        """
        for (add_row, add_col) in self.ORB_REVEAL:
            new_row = row + add_row
            new_col = col + add_col

            if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                continue

            self.board[new_row][new_col].revealed = True


    def _cast_orb_scroll(self):
        """
        Contrary to the name of the actor, this simply acts like an orb. However, it chooses
        a spot to reveal, rather than simply on itself. Note that this function is taken
        straight from the game's code. It reveals up to 1.5 Euclidean distance away. So,
        smaller than the orb.
        """
        candidates = []
        choice = None

        # Get all none revealed squares
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if not self.board[row][col].revealed:
                    candidates.append((row, col))

        # Shuffle Candidates
        random.shuffle(candidates)

        # Check choose first candidate close to a mine as the choice
        for candidate in candidates:
            surrounding = self.get_surrounding_cells(candidate, True)
            for spot in surrounding:
                if self.board[spot[0]][spot[1]].actor == Actors.MINE:
                    choice = candidate
                    break
            if choice is not None:
                break

        # If none are close to a mine, simply pick the first candidate (if possible)
        if choice is None and len(candidates) > 0:
            choice = candidates[0]

        # If still no candidates, do nothing
        if choice is None:
            return

        # Reveal area (3 x 3 surrounding grid) of the choice (and center)
        self.board[choice[0]][choice[1]].revealed = True
        for (row, col) in self.get_surrounding_cells(choice, True):
            self.board[row][col].revealed = True


    def _update_surrounding_cells(self, row, col, power):
        """
        Reduces all cells surrounding (row, col) by the amount indicated by power.
        Intended to be called when the enemy occupying (row, col) is killed.

        :param row: The row of the cell
        :param col: The col of the cell
        :param power: The power of the cell before death
        """
        for (ajd_row, ajd_col) in self.get_surrounding_cells((row, col), True):
            self.board[ajd_row][ajd_col].adj_power -= power


    @staticmethod
    def get_surrounding_cells(cell, diagonal):
        """
        Get all surrounded cells on the board.

        :param cell: The center cell from which to return the surrounding cells
        :param diagonal: If true, return cells diagonal to the center cell, otherwise don't
        :return:
        """
        output = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                # Skip if center cell
                if i == j == 0:
                    continue

                # If not diagonal, skip diagonal additions
                if not diagonal and i != 0 and j != 0:
                    continue

                row = cell[0] + i
                col = cell[1] + j

                # Skip if outside bounds
                if 0 > row or row > 9 or 0 > col or col > 12:
                    continue

                output.append((row, col))

        random.shuffle(output)
        return output


    def __str__(self):
        """
        Give a string representation of the full board, as well as what the agent can see.

        :return:
        """
        bg_even = '\033[48;5;254m'
        reset = "\033[0m"

        output = '\n' * 20
        output += "-" * 197 + '\n'
        output += "                                                                             FULL BOARD \n"
        output += "-" * 197 + '\n'
        output += "       0        |       1      |       2      |       3      |       4      |       5      |       6      |       7      |       8      |       9      |      10      |      11      |      12      |\n"
        output += "                |              |              |              |              |              |              |              |              |              |              |              |              |\n"
        output += "0  "

        row_counter = 1
        for i in range(self.ROWS):
            for j in range(self.COLS):
                cell_str = ""
                cell = self.board[i][j]
                if cell.actor == Actors.MINE:
                    if cell.power == 0:
                        cell_str += "MINE      (0)| "
                    else:
                        cell_str += "MINE    (100)| "
                elif cell.actor == Actors.EMPTY:
                    cell_str += "   " + "[" + str(cell.adj_power) + "]" + ("     | " if cell.adj_power >= 100 else "      | " if cell.adj_power >= 10 else "       | ")
                else:
                    cell_str += self._actor_to_str(cell.actor, cell.xp) + f"({cell.power})" + (" | " if cell.power < 10 else "| ")

                if i % 2 == 0:
                    output += f"{bg_even}{cell_str}{reset}"
                else:
                    output += cell_str

            output += '\n'
            if row_counter < 10:
                output += str(row_counter) + ("  " if row_counter < 10 else " ")
                row_counter += 1

        output += "-" * 197 + '\n'
        output += "                                                                             GAME BOARD \n"
        output += "-" * 197 + '\n'
        output += "       0        |       1      |       2      |       3      |       4      |       5      |       6      |       7      |       8      |       9      |      10      |      11      |      12      |\n"
        output += "                |              |              |              |              |              |              |              |              |              |              |              |              |\n"
        output += "0  "

        row_counter = 1
        for i in range(self.ROWS):
            for j in range(self.COLS):
                cell_str = ""
                cell = self.board[i][j]
                if not cell.revealed:
                    cell_str +=  "UNKNOWN      | "
                elif cell.actor == Actors.MINE:
                    cell_str += f"MINE  (100)  | "
                elif cell.actor == Actors.MIMIC:
                    cell_str += f"CHEST        | "
                elif cell.actor == Actors.EMPTY:
                    if cell.obscured:
                        cell_str += f"    ???      | "
                    else:
                        cell_str += f"[{str(cell.adj_power)}]" + (" " * 8 if cell.adj_power >= 100 else " " * 9 if cell.adj_power >= 10 else " " * 10) + "| "
                else:
                    cell_str += self._actor_to_str(cell.actor, cell.xp) + "(" + str(cell.power) + ")" + (" | " if cell.power < 10 else "| ")

                if i % 2 == 0:
                    output += f"{bg_even}{cell_str}{reset}"
                else:
                    output += cell_str

            output += '\n'
            if row_counter < 10:
                output +=  str(row_counter) + ("  " if row_counter < 10 else " ")
                row_counter += 1

        curr_level = self.level - 1 if self.level < len(self.XP_REQUIREMENTS) else len(self.XP_REQUIREMENTS) - 1
        output += "-" * 197 + '\n'
        output += f"                                                                 {self.curr_health}/{self.max_health} HP  |  {self.xp}/{self.XP_REQUIREMENTS[curr_level]} XP  |  (Score: {self.score})\n"
        output += "-" * 197 + '\n'


        return output

    @staticmethod
    def _actor_to_str(actor, xp_val):
        if actor == Actors.EMPTY:
            return "         "
        elif actor == Actors.DRAGON:
            return "DRAGON   "
        elif actor == Actors.CROWN:
            return "CROWN    "
        elif actor == Actors.DRAGON_EGG:
            return "EGG      "
        elif actor == Actors.WIZARD:
            return "WIZARD   "
        elif actor == Actors.BIG_SLIME:
            return "B. SLIME "
        elif actor == Actors.MINE_KING:
            return "M. KING  "
        elif actor == Actors.GIANT:
            return "GIANT    "
        elif actor == Actors.CHEST:
            return "CHEST    "
        elif actor == Actors.MINOTAUR:
            return "MINOTAUR "
        elif actor == Actors.WALL:
            return "WALL     "
        elif actor == Actors.GUARD:
            return "GUARDIAN "
        elif actor == Actors.GARGOYLE:
            return "GARGOYLE "
        elif actor == Actors.GAZER:
            return "GAZER    "
        elif actor == Actors.MEDIKIT:
            return "MEDIKIT  "
        elif actor == Actors.GNOME:
            return "GNOME    "
        elif actor == Actors.MINE:
            return "MINE     "
        elif actor == Actors.ORB:
            return "ORB      "
        elif actor == Actors.RAT:
            return "RAT      "
        elif actor == Actors.BAT:
            return "BAT      "
        elif actor == Actors.SKELETON:
            return "SKELETON "
        elif actor == Actors.SLIME:
            return "SLIME    "
        elif actor == Actors.MIMIC:
            return "MIMIC    "
        elif actor == Actors.SPELL_MAKE_ORB:
            return "MK. ORB  "
        elif actor == Actors.XP:
            return f"{xp_val}XP" + (" " * 6 if xp_val < 10 else " " * 5)
        elif actor == Actors.SPELL_DISARM:
            return "S. DISARM"
        elif actor == Actors.SPELL_REVEAL_RATS:
            return "SP. RATS "
        elif actor == Actors.SPELL_REVEAL_SLIMES:
            return "SP. SLIME"
        else:
            return "ERROR    "

if __name__ == "__main__":
    game = Game()
    game.reset_game()

    alive = True
    terminated = False

    while True:
        print(game)
        choice = input("Touch a square (1) or level up (2) > ")

        if choice == "1":
            row = input("Row > ")
            col = input("Col > ")
            if not row.isdigit() or not col.isdigit():
                input("Invalid choice! Must be an integer. Press enter to continue")
            else:
                row = int(row)
                col = int(col)
                if row < 0 or row > 9 or col < 0 or col > 12:
                    input("Invalid choice! Values outside of range. Press enter to continue")
                else:
                    alive, terminated = game.touch_square(row, col)
            if alive and terminated:
                print(game)
                print(f"Game Over! Won with a score of {game.score}/365")
                exit(0)
            elif not alive:
                print(game)
                print("Game Over! You Died!")
                exit(0)

        elif choice == "2":
            success = game.level_up()
            if success:
                input("You successfully levelled up! Press enter to continue")
            else:
                input("You do not have enough XP to level up. Press enter to continue")

        else:
            input("Invalid choice! Press enter to continue")
