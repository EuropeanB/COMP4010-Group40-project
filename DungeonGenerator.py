from Cell import Cell
from Actors import Actors
import math
import random


class DungeonGenerator:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.DRAGON_POSITION = (4, 6)  # Dragon always at (4,6)
        self.LEGAL_EGG_POSITIONS = [  # Dragon Egg always one away from Dragon
            (3, 5), (3, 6), (3, 7), (4, 5), (4, 7), (5, 5), (5, 6), (5, 7)
        ]
        self.EDGE_SPOTS = ( # Doesn't include corners
                [(0, col) for col in range(1, 12)] +
                [(row, 0) for row in range(1, 9)] +
                [(9, col) for col in range(1, 12)] +
                [(row, 12) for row in range(1, 9)]
        )
        self.CORNERS = [(0, 0), (0, 12), (9, 0), (9, 12)]

        self.ORB_REVEAL = (
            [
                (-2, 0),
                (-1, -1), (-1, 0), (-1, 1),
                (0, -2), (0, -1), (0, 1), (0, 2),
                (1, -1), (1, 0), (1, 1),
                (2, 0)
            ]
        )

        self.rem = None
        self.board = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]


    @staticmethod
    def distance(cell1, cell2):
        """
        Helper function that calculates Euclidean distance two points on the grid

        :param cell1: (row, col) tuple for the first cell
        :param cell2: (row, col) tuple for the second cell
        :return: Euclidean distance between the two points
        """
        return math.sqrt((cell2[1] - cell1[1]) ** 2 + (cell2[0] - cell1[0]) ** 2)


    def get_surrounding_cells(self, cell, diagonal):
        """
        Get all surrounded cells that are remaining options on the board.

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

                # Check if spot taken
                if (row, col) not in self.rem:
                    continue

                output.append((row, col))

        random.shuffle(output)
        return output


    def generate_dungeon(self):
        # Reset Cells (Required because of bug issues)
        self.board = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]

        # Reset Remaining Positions
        self.rem = [(row, col) for row in range(10) for col in range(13)]
        random.shuffle(self.rem)

        # Place actors
        self._place_dragon_and_egg() # Cannot Fail
        wizard_location = self._place_wizard_and_big_slimes() # Cannot Fail
        self._place_mine_king(wizard_location) # Cannot Fail
        if not self._place_giants():
            return False
        if not self._place_minotaurs_and_chests():
            return False
        if not self._place_walls():
            return False
        if not self._place_guardians():
            return False
        if not self._place_gargoyles():
            return False
        if not self._place_medikits_and_gnome():
            return False
        if not self._place_gazers():
            return False
        if not self._place_mines():
            return False
        if not self._place_orb(): # Must be after Medikit, Wall, Dragon, Gazer, Chest, Mimic, Make Orb, Rat King, Mine, and Big Slime
            return False
        if not self._place_minor_monsters():
            return False

        # Calculate surrounding power for all cells
        for row in range(self.rows):
            for col in range(self.cols):
                self.board[row][col].adj_power = self._calculate_surrounding_power(row, col)

        return self.board


    def _place_dragon_and_egg(self):
        """
        Places the Dragon dead center and the Dragon Egg nearby
        """
        self.board[self.DRAGON_POSITION[0]][self.DRAGON_POSITION[1]].actor = Actors.DRAGON
        self.board[self.DRAGON_POSITION[0]][self.DRAGON_POSITION[1]].power = 13
        self.board[self.DRAGON_POSITION[0]][self.DRAGON_POSITION[1]].revealed = True
        self.rem.remove((self.DRAGON_POSITION[0], self.DRAGON_POSITION[1]))
        egg_row, egg_col = random.choice(self.LEGAL_EGG_POSITIONS)
        self.board[egg_row][egg_col].actor = Actors.DRAGON_EGG
        self.rem.remove((egg_row, egg_col))


    def _place_wizard_and_big_slimes(self):
        """
        Places Wizard on the edge and surround them with Big Slimes

        :return: The location of the Wizard for use in other functions
        """
        # Choose random spot on the edge for the wizard
        wizard_spot = random.choice(self.EDGE_SPOTS)
        wizard_row, wizard_col = wizard_spot
        self.board[wizard_row][wizard_col].actor = Actors.WIZARD
        self.board[wizard_row][wizard_col].power = 1
        self.rem.remove((wizard_row, wizard_col))

        # Surround the wizard with slimes
        big_slime_spots = self.get_surrounding_cells(wizard_spot, True)
        for spot in big_slime_spots:
            self.board[spot[0]][spot[1]].actor = Actors.BIG_SLIME
            self.board[spot[0]][spot[1]].power = 8
            self.rem.remove(spot)

        return wizard_spot


    def _place_mine_king(self, wizard_location):
        """
        Places the Mine King in a corner. The Wizard's location is required to make sure
        the Mine King doesn't overwrite a Big Slime (if Wizard is near a corner)

        :param wizard_location: The (row, col) coordinates of the wizard
        """
        mine_king_options = self.CORNERS.copy()
        if wizard_location in [(0, 1), (1, 0)]:  # Near Top Left Corner
            mine_king_options.remove((0, 0))
        elif wizard_location in [(8, 0), (9, 1)]:  # Near Bottom Left Corner
            mine_king_options.remove((9, 0))
        elif wizard_location in [(0, 11), (1, 12)]:  # Near Top Right Corner
            mine_king_options.remove((0, 12))
        elif wizard_location in [(8, 12), (9, 11)]:  # Near Bottom Right Corner
            mine_king_options.remove((9, 12))

        mine_king_row, mine_king_col = random.choice(mine_king_options)
        self.board[mine_king_row][mine_king_col].actor = Actors.MINE_KING
        self.board[mine_king_row][mine_king_col].power = 10
        self.rem.remove((mine_king_row, mine_king_col))


    def _place_giants(self):
        """
        Both Giant need to be placed such that they are in the same row, and same distance from the center

        :return: True if successful, False otherwise
        """
        # Loop until good position found
        for (giant_row, giant_col) in [(row, col) for row, col in self.rem if col < 6]:
            giant_col2 = self.cols - 1 - giant_col

            if (giant_row, giant_col2) in self.rem:
                self.board[giant_row][giant_col].actor = Actors.GIANT
                self.board[giant_row][giant_col].power = 9
                self.board[giant_row][giant_col2].actor = Actors.GIANT
                self.board[giant_row][giant_col2].power = 9
                self.rem.remove((giant_row, giant_col))
                self.rem.remove((giant_row, giant_col2))
                return True

        return False


    def _place_minotaurs_and_chests(self):
        """
        Start by placing Minotaurs. Each Minotaur is assigned a chest. The chests must be at least 3 Euclidean
        distance from each other. Currently, this function works but restarts the entire board if it fails too
        many times. This will be updated.

        :return: True if successful, False otherwise
        """
        NUM_MINOTAURS = 5

        minotaur_indices = [0] * NUM_MINOTAURS
        chest_locations = []

        i = 0
        while i < NUM_MINOTAURS:
            # If index goes out of bounds, then current setup is infeasible, and should fall to previous index
            if minotaur_indices[i] >= len(self.rem):
                # If first index tries to fall back, then setup cannot be created (restart)
                if i == 0:
                    return False
                chest_locations.pop()
                i -= 1
                continue

            # Get all possible chest locations for the current Minotaur
            possible_chests = self.get_surrounding_cells(self.rem[minotaur_indices[i]], True)

            # Loop through all possible locations
            for curr_chest in possible_chests:
                qualified = True

                # Check if chest is far enough from all other chests
                for chest in chest_locations:
                    if self.distance(curr_chest, chest) < 3.0:
                        qualified = False
                        break

                # If chest qualifies, add it to the list of chests and move on
                if qualified:
                    chest_locations.append(curr_chest)
                    break

            # If a chest was not added, current Minotaur location was not valid and we should try next
            if len(chest_locations) == i:
                minotaur_indices[i] += 1
            else:
                i += 1
                if i < NUM_MINOTAURS:
                    minotaur_indices[i] = minotaur_indices[i - 1] + 1

        # Need to loop from high to low, because removing values will change index order
        for i in range(NUM_MINOTAURS - 1, -1, -1):
            minotaur_row, minotaur_col = self.rem[minotaur_indices[i]]
            chest_row, chest_col = chest_locations[i]
            self.board[minotaur_row][minotaur_col].actor = Actors.MINOTAUR
            self.board[minotaur_row][minotaur_col].power = 6
            self.board[chest_row][chest_col].actor = Actors.CHEST
            if i > 2: # 2 Contain Medikit (The rest contain 5 XP, handled in main loop)
                self.board[chest_row][chest_col].contains_medikit = True
            self.rem.remove((minotaur_row, minotaur_col))
            try:
                self.rem.remove((chest_row, chest_col))
            except:
                return False

        return True


    def _place_walls(self):
        """
        Walls must come in pairs (directly next to each other, not diagonally). Additionally, both Walls
        in a pairing cannot be against the edge (however, one may). Lastly, pairs of walls cannot be nearby
        to another pair of walls

        :return" True if successful, False otherwise
        """
        # Loop through possible first pair, first wall spots
        EDGE_AND_CORNERS = self.EDGE_SPOTS + [(0, 0), (0, 12), (9, 0), (9, 12)]

        for wall_one_one in self.rem:
            # Get surrounding legal cells
            surrounding_one = self.get_surrounding_cells(wall_one_one, False)
            edge_spot_one = wall_one_one in EDGE_AND_CORNERS

            # Check possible first pair, second wall spots
            for wall_one_two in surrounding_one:

                # Make sure both aren't wall spots
                if edge_spot_one and wall_one_two in EDGE_AND_CORNERS:
                    continue

                # Loop through possible second pair, first wall spots
                for wall_two_one in [(x, y) for (x, y) in self.rem if (x, y) not in [wall_one_one, wall_one_two]]:

                    # Make sure distance at least 2 away other walls
                    if self.distance(wall_two_one, wall_one_one) < 2 or self.distance(wall_two_one, wall_one_two) < 2:
                        continue

                    # Get surrounding legal cells
                    surrounding_two = self.get_surrounding_cells(wall_two_one, False)
                    edge_spot_two = wall_two_one in EDGE_AND_CORNERS

                    # Check possible second pair, second wall spots
                    for wall_two_two in surrounding_two:

                        # Make sure both aren't wall spots
                        if edge_spot_two and wall_two_two in EDGE_AND_CORNERS:
                            continue

                        # Loop through possible third pair, first wall spots
                        for wall_three_one in [(x, y) for (x, y) in self.rem if
                                               (x, y) not in [wall_one_one, wall_one_two, wall_two_one, wall_two_two]]:

                            # Make sure distance at least 2 away from other walls
                            if (self.distance(wall_three_one, wall_two_one) < 2 or self.distance(wall_three_one,
                                                                                                 wall_two_two) < 2 or
                                    self.distance(wall_three_one, wall_one_one) < 2 or self.distance(wall_three_one,
                                                                                                     wall_one_two) < 2):
                                continue

                            # Get surrounding legal cells
                            surrounding_three = self.get_surrounding_cells(wall_three_one, False)
                            edge_spot_three = wall_three_one in EDGE_AND_CORNERS

                            # Check possible third pair, second wall spots
                            for wall_three_two in surrounding_three:

                                # Make sure both aren't wall spots
                                if edge_spot_three and wall_three_two in EDGE_AND_CORNERS:
                                    continue

                                # Update board and remove from remaining
                                self.board[wall_one_one[0]][wall_one_one[1]].actor = Actors.WALL
                                self.board[wall_one_one[0]][wall_one_one[1]].power = 3
                                self.board[wall_one_two[0]][wall_one_two[1]].actor = Actors.WALL
                                self.board[wall_one_two[0]][wall_one_two[1]].power = 3
                                self.board[wall_two_one[0]][wall_two_one[1]].actor = Actors.WALL
                                self.board[wall_two_one[0]][wall_two_one[1]].power = 3
                                self.board[wall_two_two[0]][wall_two_two[1]].actor = Actors.WALL
                                self.board[wall_two_two[0]][wall_two_two[1]].power = 3
                                self.board[wall_three_one[0]][wall_three_one[1]].actor = Actors.WALL
                                self.board[wall_three_one[0]][wall_three_one[1]].power = 3
                                self.board[wall_three_two[0]][wall_three_two[1]].actor = Actors.WALL
                                self.board[wall_three_two[0]][wall_three_two[1]].power = 3
                                self.rem.remove(wall_one_one)
                                self.rem.remove(wall_one_two)
                                self.rem.remove(wall_two_one)
                                self.rem.remove(wall_two_two)
                                self.rem.remove(wall_three_one)
                                self.rem.remove(wall_three_two)

                                return True

        return False


    def _place_guardians(self):
        """
        The only rule is to have a single guardian in each quadrant. Note that the quadrants are defined
        by the row and col where the dragon is positioned.

        :return: True if successful, False otherwise
        """
        row_limit, col_limit = self.DRAGON_POSITION

        # Choose a random spot from eaech quadrant
        try:
            first_guardian = random.choice([(row, col) for (row, col) in self.rem if row > row_limit and col < col_limit])
            second_guardian = random.choice([(row, col) for (row, col) in self.rem if row < row_limit and col < col_limit])
            third_guardian = random.choice([(row, col) for (row, col) in self.rem if row > row_limit and col > col_limit])
            fourth_guardian = random.choice([(row, col) for (row, col) in self.rem if row < row_limit and col > col_limit])

            self.board[first_guardian[0]][first_guardian[1]].actor = Actors.GUARD
            self.board[first_guardian[0]][first_guardian[1]].power = 7
            self.board[second_guardian[0]][second_guardian[1]].actor = Actors.GUARD
            self.board[second_guardian[0]][second_guardian[1]].power = 7
            self.board[third_guardian[0]][third_guardian[1]].actor = Actors.GUARD
            self.board[third_guardian[0]][third_guardian[1]].power = 7
            self.board[fourth_guardian[0]][fourth_guardian[1]].actor = Actors.GUARD
            self.board[fourth_guardian[0]][fourth_guardian[1]].power = 7
            self.rem.remove(first_guardian)
            self.rem.remove(second_guardian)
            self.rem.remove(third_guardian)
            self.rem.remove(fourth_guardian)

            return True

        # If ever there's no random choice to be made, then the board is infeasible
        except IndexError:
            return False


    def _place_gargoyles(self):
        """
        Each gargoyle must be placed next to another gargoyle (not diagonally)

        :return: True if successful, False otherwise
        """
        NUM_GARGOYLES_PAIRS = 4
        num_added = 0

        for i in range(NUM_GARGOYLES_PAIRS):
            # Loop through possible first spots for the gargoyle
            for first_spot in self.rem:
                # Get all surrounding spots (minus diagonals)
                possible_second_spots = self.get_surrounding_cells(first_spot, False)

                # If there's a legal spot, add it
                if len(possible_second_spots) > 0:
                    second_spot = possible_second_spots[0]
                    self.board[first_spot[0]][first_spot[1]].actor = Actors.GARGOYLE
                    self.board[first_spot[0]][first_spot[1]].power = 4
                    self.board[second_spot[0]][second_spot[1]].actor = Actors.GARGOYLE
                    self.board[second_spot[0]][second_spot[1]].power = 4
                    self.rem.remove(first_spot)
                    self.rem.remove(second_spot)
                    num_added += 1
                    break

        return num_added == NUM_GARGOYLES_PAIRS


    def _place_medikits_and_gnome(self):
        """
        Medikits must be placed 3.5 Euclidean distance from each other. Gnome is placed next to any medikit

        :return: True if successful, False otherwise
        """
        NUM_MEDIKITS = 5

        medikit_indices = [0] * 5

        i = 0
        while i < NUM_MEDIKITS:
            # If index goes out of bounds, then current setup is infeasible, and should fall to previous index
            if medikit_indices[i] >= len(self.rem):
                # If first index tries to fall back, then setup cannot be created (restart)
                if i == 0:
                    return False
                i -= 1
                continue

            # Ensure Medikit is far enough from all other medikits
            qualified = True
            for j in range(i):
                if self.distance(self.rem[medikit_indices[i]], self.rem[medikit_indices[j]]) < 3.5:
                    qualified = False
                    break

            # If it qualifies, move to next index and start it at one above current
            if qualified:
                i += 1
                if i < NUM_MEDIKITS:
                    medikit_indices[i] = medikit_indices[i - 1] + 1
            # Otherwise, check next option
            else:
                medikit_indices[i] += 1

        # Update board in reverse order, as removing will update indices
        gnome_placed = False
        for i in range(NUM_MEDIKITS - 1, -1, -1):
            medikit_row, medikit_col = self.rem[medikit_indices[i]]

            # Place Gnome if possible and not done yet
            if not gnome_placed:
                possible_spots = self.get_surrounding_cells((medikit_row, medikit_col), True)
                if len(possible_spots) > 0:
                    gnome_placed = True
                    gnome_row, gnome_col = possible_spots[0]
                    self.board[gnome_row][gnome_col].actor = Actors.GNOME
                    self.rem.remove((gnome_row, gnome_col))

            # Update board for Medikit
            self.board[medikit_row][medikit_col].actor = Actors.MEDIKIT
            self.rem.remove((medikit_row, medikit_col))

        return gnome_placed


    def _place_gazers(self):
        """
        No rules for placing gazers, just randomly. Gazers do blind in a 2.0 Euclidean distance circle, however.

        :return: True if successful, False otherwise
        """
        NUM_GAZERS = 2

        for i in range(NUM_GAZERS):
            if len(self.rem) == 0:
                return False

            gazer_spot = self.rem[0]
            self.board[gazer_spot[0]][gazer_spot[1]].actor = Actors.GAZER
            self.board[gazer_spot[0]][gazer_spot[1]].power = 5
            self.rem.remove(gazer_spot)

            # Obscure Surrounding Squares (Similar to Orb Reveal)
            for (add_row, add_col) in self.ORB_REVEAL:
                new_row = gazer_spot[0] + add_row
                new_col = gazer_spot[1] + add_col

                if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
                    continue

                self.board[new_row][new_col].obscured = True

        return True


    def _place_mines(self):
        """
        No rules for placing mines, simply place them randomly.

        :return: True if successful, False otherwise
        """
        NUM_MINES = 9

        for i in range(NUM_MINES):
            if len(self.rem) == 0:
                return False

            mine_spot = self.rem[0]
            self.board[mine_spot[0]][mine_spot[1]].actor = Actors.MINE
            self.board[mine_spot[0]][mine_spot[1]].power = 100
            self.rem.remove(mine_spot)

        return True


    def _place_minor_monsters(self):
        """
        Place the remaining monsters: 13 Rats, 12 Bats, 10 Skeletons, 8 Slimes, 1 Mimic, 1 Spell Orb

        :return: True if successful, False otherwise
        """
        NUM_RATS = (13, Actors.RAT, 1)
        NUM_BATS = (12, Actors.BAT, 2)
        NUM_SKELETONS = (10, Actors.SKELETON, 3)
        NUM_SLIMES = (8, Actors.SLIME, 5)
        NUM_MIMICS = (1, Actors.MIMIC, 11)
        NUM_SPELL_ORBS = (1, Actors.SPELL_MAKE_ORB, 0)

        actors = [NUM_RATS, NUM_BATS, NUM_SKELETONS, NUM_SLIMES, NUM_MIMICS, NUM_SPELL_ORBS]

        for (num, actor, power) in actors:
            for _ in range(num):
                if len(self.rem) == 0:
                    return False

                spot = self.rem[0]
                self.board[spot[0]][spot[1]].actor = actor
                self.board[spot[0]][spot[1]].power = power
                self.rem.remove(spot)

        return True


    def _place_orb(self):
        """
        Place the starting orb. There are some rules surrounding this:
        - Must have two spaces between the orb and the nearest edge
        - Cannot reveal Dragon, Gazer, Chest, Make Orb Spell, Rat King,
            Mines, Dragon Egg, Big Slime, or Mimic
        - Don't want to reveal more than 2 walls. but should reveal at least 1 wall
        - Want to reveal exactly 1 medikit

        The algorithm for this is taken directly from the game code.

        :return: True if successful, False otherwise
        """
        stats = [0] * len(self.rem)

        for i in range(len(self.rem)):
            row = self.rem[i][0]
            col = self.rem[i][1]

            if row < 2 or row > (self.rows - 3) or col < 2 or col > (self.cols - 3):
                stats[i] = -10000
                continue

            num_medikits = 0
            num_walls = 0
            num_forbiddenObjects = 0

            for (add_row, add_col) in self.ORB_REVEAL:
                new_row = row + add_row
                new_col = col + add_col

                if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
                    continue

                if self.board[new_row][new_col].actor == Actors.MEDIKIT:
                    num_medikits += 1
                elif self.board[new_row][new_col].actor == Actors.WALL:
                    num_walls += 1
                elif self.board[new_row][new_col].actor in [
                    Actors.DRAGON, Actors.DRAGON_EGG, Actors.GAZER, Actors.CHEST, Actors.MIMIC,
                    Actors.SPELL_MAKE_ORB, Actors.RAT_KING, Actors.MINE, Actors.BIG_SLIME
                ]:
                    num_forbiddenObjects += 1

                stats[i] = num_forbiddenObjects * -10000 # Adjusted to 10K to really prevent forbidden objects from appearing
                stats[i] += max(0, num_walls - 2) * -2000
                stats[i] += 2000 if num_medikits == 1 and num_walls > 0 else 0

        # Get max value and verify if good placement
        max_value = max(stats)
        if max_value < 0:
            return False

        # Get argmax and update the board
        max_index = stats.index(max_value)
        row, col = self.rem[max_index]
        self.board[row][col].actor = Actors.ORB
        self.board[row][col].revealed = True
        self.rem.remove((row, col))

        return True


    def _calculate_surrounding_power(self, row, col):
        total = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                # Skip if center cell
                if i == j == 0:
                    continue

                new_row = row + i
                new_col = col + j

                # Skip if outside bounds
                if 0 > new_row or new_row > 9 or 0 > new_col or new_col > 12:
                    continue

                if self.board[new_row][new_col].actor != Actors.WALL: # Walls have power, but do not count towards adjacent power
                    total += self.board[new_row][new_col].power

        return total
