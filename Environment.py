import numpy as np
import gymnasium as gym
import time
from Game import Game
from Actors import Actors
from GameVisual import GameVisual


class DragonSweeperEnv(gym.Env):
    """
    A Gymnasium environment for DragonSweeper, a Minesweeper variant with RPG elements

    The game features:
    - Grid-based exploration with hidden enemies and items
    - Player health and experience
    - Combat mechanics where enemies cost HP but grant XP
    - Level-up system for strategic health recovery
    - Dragon boss as final objective
    """

    def __init__(self, render_mode=None):
        """
        Initialize the DragonSweeper environment

        :param render_mode: Rendering mode (not implemented but required by Gymnasium)
        """
        # Internal State
        self.game = Game()

        # Set up the rendering
        self.render_mode=render_mode
        self.game_visual = None
        if self.render_mode == "human":
            self.game_visual = GameVisual(self.game)

        # Constants
        self.ROWS = 10
        self.COLS = 13
        self.LEVEL_UP_INDEX = self.ROWS * self.COLS # The action value for levelling up
        # Actors that are safe to click
        self.SAFE_ACTORS = [Actors.ORB, Actors.SPELL_MAKE_ORB, Actors.SPELL_DISARM, Actors.SPELL_REVEAL_RATS,
                         Actors.SPELL_REVEAL_SLIMES, Actors.DRAGON_EGG, Actors.XP, Actors.GNOME, Actors.CROWN]

        # Board space indices
        self.BOARD_CHANNELS = 10

        # Cell Status Constants
        self.CELL_HIDDEN = 0
        self.CELL_REVEALED = 0.5
        self.CELL_OBSCURED = 1.0

        # Standard channels
        self.STATUS_IDX = 0
        self.ADJ_POWER_IDX = 1
        self.CELL_POWER_IDX = 2
        self.ADJ_BOMBS_IDX = 3

        # One-hot encoded channels
        self.EMPTY_IDX = 4
        self.SAFE_IDX = 5
        self.MEDIKIT_IDX = 6
        self.CHEST_IDX = 7
        self.MINE_IDX = 8 # Note, defused mines are SAFE, not MINE
        self.ENEMY_IDX = 9

        # Player space indices
        self.PLAYER_CHANNELS = 4

        # Standard channels
        self.CURRENT_HP_IDX = 0
        self.MAX_HP_IDX = 1
        self.CURRENT_XP_IDX = 2
        self.XP_REQUIRED_IDX = 3

        # Board representation constants
        self.MAX_ADJ_POWER = 40 # This isn't trivially known. This is an estimate
        self.MAX_CELL_POWER = 13 # Highest damaging enemy is the dragon at 13
        self.MAX_ADJ_BOMBS = 8

        # Board representation: Each index represents a square on the board
        # Channels: [Status (hidden 0, revealed 0.5, obscured 1), Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [EMPTY, SAFE, MEDIKIT, CHEST, MINE, ENEMY]
        # 4 channels + 6 one-hot encoding = 10 total channels
        board_space = gym.spaces.Box(
            low = np.zeros((self.ROWS, self.COLS, self.BOARD_CHANNELS), dtype=np.float32),
            high = np.ones((self.ROWS, self.COLS, self.BOARD_CHANNELS), dtype=np.float32),
            shape=(self.ROWS, self.COLS, self.BOARD_CHANNELS),
            dtype=np.float32
        )

        # Player representation constants
        self.STARTING_HP_SLOTS = 6
        self.MIN_HP = 0  # Represent all death states as 0 HP
        self.MAX_HP = 20
        self.STARTING_XP_REQUIRED = 6
        self.MAX_XP_REQUIRED = 25
        self.MIN_XP = 0
        self.MAX_XP = 30  # This isn't trivially known. This is an estimate

        # Player representation: [Current HP, HP Slots, Current XP, XP Required to Level Up]
        player_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            shape=(self.PLAYER_CHANNELS,),
            dtype=np.float32
        )

        # Combine Board and Player representations into a single observation space
        self.observation_space = gym.spaces.Dict({
            "board": board_space,
            "player": player_space
        })

        # Action space: 0-129 for selected grid cells, 130 for level-up
        self.action_space = gym.spaces.Discrete(self.ROWS * self.COLS + 1)

        # Preallocate buffers to avoid creating new arrays every step
        self._board_buffer = np.zeros((self.ROWS, self.COLS, self.BOARD_CHANNELS), dtype=np.float32)
        self._player_buffer = np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32)


    def _get_obs(self):
        """
        Translate the environment state into an observation for the agent.

        Observation Structure:
        - Board: 13 x 10 x 10 tensor with channels [Revealed, Adj. Power, Cell Power, Adj. Bombs, 6x one-hot types]
        - Player: [Current HP, Max HP, Current XP, XP Capacity]

        :return: Dictionary containing 'board' and 'player' observations
        """

        # Translate game to board space
        board_space = self._board_buffer
        board_space.fill(0)

        # Loop through every cell and translate it
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]


                # Check if cell is hidden, we have no information on it
                if not cell.revealed:
                    board_space[row, col, self.STATUS_IDX] = self.CELL_HIDDEN
                    continue


                # If the cell is revealed, get its type and one hot encode it
                actor = cell.actor
                if actor in [Actors.EMPTY, Actors.NONE]:
                    board_space[row, col, self.EMPTY_IDX] = 1.0
                elif actor in self.SAFE_ACTORS:
                    board_space[row, col, self.SAFE_IDX] = 1.0
                elif actor == Actors.MEDIKIT:
                    board_space[row, col, self.MEDIKIT_IDX] = 1.0
                elif actor == Actors.CHEST or actor == Actors.MIMIC:
                    board_space[row, col, self.CHEST_IDX] = 1.0
                elif actor == Actors.MINE:
                    # We treat defused mines as safe actions, and active mines as MINES wih 1.0 cell power
                    if cell.power == 0:
                        board_space[row, col, self.SAFE_IDX] = 1.0
                    else:
                        board_space[row, col, self.MINE_IDX] = 1.0
                        board_space[row, col, self.CELL_POWER_IDX] = 1.0
                else:
                    # Otherwise, it is an enemy and we get its power. We represent bomb power as 14
                    board_space[row, col, self.ENEMY_IDX] = 1.0
                    board_space[row, col, self.CELL_POWER_IDX] = np.float32(cell.power / self.MAX_CELL_POWER)


                # If cell is empty but obscured, we get no information
                # If cell is empty and not obscured, we get all surrounding information
                if board_space[row, col, self.EMPTY_IDX]:
                    if cell.obscured:
                        board_space[row, col, self.STATUS_IDX] = self.CELL_OBSCURED
                        continue

                    adj_bombs = cell.adj_power // 100
                    adj_power = cell.adj_power % 100

                    board_space[row, col, self.ADJ_BOMBS_IDX] = np.float32(adj_bombs / self.MAX_ADJ_BOMBS)
                    board_space[row, col, self.ADJ_POWER_IDX] = np.float32(min(1.0, adj_power / self.MAX_ADJ_POWER)) # Clip since MAX_ADJ_POWER is estimate


                # Cell is not hidden, and not obscured. So, must be revealed
                board_space[row, col, self.STATUS_IDX] = self.CELL_REVEALED


        # Translate game to player space
        player_space = self._player_buffer
        player_space[self.CURRENT_HP_IDX] = self.game.curr_health / self.MAX_HP
        player_space[self.MAX_HP_IDX] = self.game.max_health / self.MAX_HP
        player_space[self.CURRENT_XP_IDX] = min(1.0, self.game.xp / self.MAX_XP) # Clip since MAX_XP is estimate, but should never really be that high anyway
        player_space[self.XP_REQUIRED_IDX] = self.game.get_required_level_xp() / self.MAX_XP_REQUIRED

        # Return Observation
        return {"board": board_space, "player": player_space}


    def _get_info(self):
        """
        Returns diagnostic information for debugging/monitoring. Currently, returns
        the score if the game is won, or the cause of death if lost.

        :param action: The action that was taken
        :param terminated: If the action led to a terminal state
        :param alive: If the action led to the player dying
        :return: Auxiliary information associated to the current state
        """
        return {
            "score": self.game.score,
            "last touched": "None" if self.game.last_touched is None else self.game.last_touched.name,
            "hp": self.game.curr_health,
            "max hp": self.game.max_health,
            "xp": self.game.xp,
            "required xp": self.game.get_required_level_xp(),
            "level": self.game.level
        }


    def get_legal_moves_mask(self, state):
        """
        Given an observation state, the environment will generate a legal
        move mask to apply to the generated q-values.

        :param state:
        :return:
        """
        legal_mask = np.zeros(131, dtype=bool)

        # Level-up action
        legal_mask[self.LEVEL_UP_INDEX] = (state['player'][self.CURRENT_XP_IDX] >= state['player'][self.XP_REQUIRED_IDX])
        # Print a message if level-up is legal
        # if legal_mask[self.LEVEL_UP_INDEX]:
        #     print("Level-up action is legal.")

        # Board actions - Illegal if revealed AND empty
        revealed = (state['board'][:, :, self.STATUS_IDX] == self.CELL_REVEALED) | (state['board'][:, :, self.STATUS_IDX] == self.CELL_OBSCURED)
        empty = state['board'][:, :, self.EMPTY_IDX] > 0.5
        legal_mask[:self.LEVEL_UP_INDEX] = (~(revealed & empty)).flatten()

        return legal_mask


    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        :param seed: Random seed for reproducibility
        :param options: Additional options for environment reset
        :return: Tuple of (observation, info) for the initial state
        """
        super().reset(seed=seed)

        # Reset game
        self.game.reset_game(seed=seed)

        return self._get_obs(), self._get_info()


    # Convert an action to a board position *assuming it can be converted*
    # This function deliberately doesn't have a check for levelling up
    # Since such a guard would force everyone to include a potentially superfluous if
    def _action_pos(self, action: int):
        ROW = action // self.COLS
        COL = action % self.COLS
        return ROW, COL


    def _calculate_reward(self, old_board, action, win, alive, success, prev_hp,
                        revealed_new_tiles_count=0,
                        killed_monster=False,
                        picked_loot=False,
                        leveled_up=False):


        # ---------------------------------------------------------------------
        # === SHAPING CONSTANTS (tune these) ===
        SURVIVAL_BONUS = 0.50       #originally 0.02
        REVEAL_BONUS = 0.35         #originally 0.12
        KILL_BONUS = 0.35           #originally 0.35
        LOOT_BONUS = 0.80           #originally 0.25
        LEVEL_BONUS = 0.90          #originally 0.60
        DEATH_PENALTY = -5.00       #originally -2.0
        # ---------------------------------------------------------------------

        # If nonsense move: heavy penalty (should never happen)
        if not success:
            return -1.0 # originally -1.0

        # Agent died
        if not alive:
            return DEATH_PENALTY

        # Winning the game
        if win:
            print("AGENT WON")
            return 1.0 + LEVEL_BONUS  # small boost

        # Compute original event reward (base_reward)
        row, col = self._action_pos(action)

        # ========= YOUR ORIGINAL LOGIC — PRESERVED, VALUES CHANGED WHERE NEEDED ========= #

        # LEVEL UP or MEDIKIT tile
        if action == self.LEVEL_UP_INDEX or old_board[row, col, self.MEDIKIT_IDX]:
            if prev_hp == 1:
                base_reward = 1.0      # originally 1.0
            elif prev_hp == 2:
                base_reward = 0.1      # originally 0.1
            else:
                base_reward = -0.4     # originally -0.4
        # SAFE tile
        elif old_board[row, col, self.SAFE_IDX]:
            base_reward = 1.0          # originally 1.0
        # CHEST tile (good or mimic survived)
        elif old_board[row, col, self.CHEST_IDX]:
            base_reward = 0.6          # originally 0.6
        # Clicking a revealed tile
        elif old_board[row, col, self.STATUS_IDX] == self.CELL_REVEALED:
            base_reward = 0.3          # originally 0.3
        else:
            # GUESSING LOGIC
            adj_revealed = False
            for row_sum in (-1, 0, 1):
                for col_sum in (-1, 0, 1):
                    if row_sum == col_sum == 0:
                        continue
                    new_row = row + row_sum
                    new_col = col + col_sum
                    if not(0 <= new_row < self.ROWS and 0 <= new_col < self.COLS):
                        continue

                    revealed = old_board[new_row, new_col, self.STATUS_IDX] == self.CELL_REVEALED
                    empty    = old_board[new_row, new_col, self.EMPTY_IDX]
                    adj_power = old_board[new_row, new_col, self.ADJ_POWER_IDX]
                    adj_mines = old_board[new_row, new_col, self.ADJ_BOMBS_IDX]

                    if not (revealed and empty):
                        continue

                    adj_revealed = True

                    # Safe to click (0-power)
                    if adj_power == 0 and adj_mines == 0:
                        base_reward = 1.0   # originally 0.8
                        break

                    # Safe-ish based on HP
                    if adj_power < prev_hp and adj_mines == 0:
                        base_reward = 0.3  # originally 0.3
                        break

            else:
                # No break occurred => adj_revealed is still False
                if not adj_revealed:
                    base_reward = -0.1     # originally -0.7
                else:
                    base_reward = 0.0      # unchanged

        # ========================================================================= #
        # ------------------------- APPLY SHAPING REWARDS ------------------------- #
        shaped = base_reward

        # 1) Survival bonus (non-death step)
        shaped += SURVIVAL_BONUS

        # 2) Reveal tiles bonus
        if revealed_new_tiles_count > 0:
            shaped += REVEAL_BONUS * revealed_new_tiles_count

        # 3) Combat rewards
        if killed_monster:
            shaped += KILL_BONUS

        # 4) Loot reward
        if picked_loot:
            shaped += LOOT_BONUS

        # 5) Level up reward
        if leveled_up:
            shaped += LEVEL_BONUS

        return float(shaped)


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        old_board = self._get_obs()['board']

        prev_hp = self.game.curr_health

        # Take action and check termination
        # Success is true if action did something, false otherwise
        if action == self.LEVEL_UP_INDEX:
            alive = True
            win = False
            success = self.game.level_up()
        else:
            row, col = self._action_pos(action)
            alive, win, success = self.game.touch_square(row, col)

        # Check termination
        terminated = not alive or win

        # Calculate reward
        reward = self._calculate_reward(old_board, action, win, alive, success, prev_hp)

        # Get observation (IMPORTANT THAT THIS IS AFTER REWARD
        observation = self._get_obs()

        # Get Truncated
        truncated = False

        # Get Info
        info = self._get_info()

        # Update render if required
        if self.render_mode == "human" and self.game_visual:
            self.render()

        return observation, reward, terminated, truncated, info


    def render(self, delay=0.6):
        """
        Render the environment.
        """
        if self.render_mode != "human" or not self.game_visual:
            return
        self.game_visual.update_display()
        time.sleep(delay)


    def close(self):
        """
        Close the Pygame window and clean up resources.
        """
        if self.game_visual:
            self.game_visual.close()
            self.game_visual = None