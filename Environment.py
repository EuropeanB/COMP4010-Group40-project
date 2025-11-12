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

        # Reward function constants
        self.REWARD_WIN = 1.0 # Reward for winning the game
        self.REWARD_SAFE_SQUARE = 0.8 # Reward for clicking on a safe square
        self.REWARD_PERFECT_LEVEL_OR_HEAL = 0.7 # Reward for optimally levelling up or healing
        self.REWARD_GUARANTEED_SAFE = 0.6 # Reward for clicking a hidden square that is deduced to be safe
        self.REWARD_CHEST_MIMIC = 0.4 # Reward for clicking a chest or killing a mimic
        self.REWARD_OPTIMAL_HP_KILL = 0.3 # Reward for killing a revealed enemy equal to our HP value
        self.REWARD_NON_OPTIMAL_HP_KILL = 0.2 # Reward for killing a revealed enemy not equal to our HP value
        self.REWARD_GUARANTEED_NO_DEATH = 0.15 # Reward for clicking a hidden square that is deduced to not kill us
        self.REWARD_GOOD_LEVEL_OR_HEAL = 0.05 # Reward for levelling up or healing slightly non-optimally
        self.REWARD_BAD_LEVEL_OR_HEAL = -0.1 # Reward for levelling up or healing poorly
        self.REWARD_RANDOM_GUESS = -0.2 # Reward for simply guessing a square with no information
        self.REWARD_DEATH = -0.3 # Reward for dying
        self.REWARD_NONSENSE = -0.5 # Reward for making a move that does nothing
        self.REWARD_OTHER_ACTIONS = 0.01 # This shouldn't ever happen

        # If a cell's value isn't known, it is set to this value:
        self.UNKNOWN_VALUE = -1.0
        self.LEGAL_VALUE = 1.0
        self.ILLEGAL_VALUE = 0.0

        # Board space indices
        self.BOARD_CHANNELS = 7

        # Standard channels
        self.ADJ_POWER_IDX = 0
        self.CELL_POWER_IDX = 1
        self.ADJ_BOMBS_IDX = 2

        # One-hot encoded channels
        self.EMPTY_IDX = 3
        self.MEDIKIT_IDX = 4
        self.CHEST_IDX = 5
        self.WALL_IDX = 6

        # Player space indices
        self.PLAYER_CHANNELS = 4

        # Standard channels
        self.CURRENT_HP_IDX = 0
        self.MAX_HP_IDX = 1
        self.CURRENT_XP_IDX = 2
        self.XP_REQUIRED_IDX = 3

        # Board representation constants
        self.MAX_ADJ_POWER = 40 # This isn't trivially known. This is an estimate
        self.MAX_CELL_POWER = 20 # Highest damaging enemy is the dragon at 13 but we represent mines as 20 (instant kill, max hp)
        self.MAX_ADJ_BOMBS = 8

        # Board representation: Each index represents a square on the board
        # Channels: [Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [EMPTY, MEDIKIT, CHEST, WALL]
        # 3 channels + 4 one-hot encoding = 7 total channels
        low_board_vals = np.full((self.BOARD_CHANNELS, self.ROWS, self.COLS), -1.0, dtype=np.float32)
        low_board_vals[3:, :, :] = 0
        high_board_vals = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32)

        board_space = gym.spaces.Box(
            low = low_board_vals,
            high = high_board_vals,
            shape=(self.BOARD_CHANNELS, self.ROWS, self.COLS),
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
        low_player_vals = np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32)
        high_player_vals = np.ones((self.PLAYER_CHANNELS,),  dtype=np.float32)
        player_space = gym.spaces.Box(
            low=low_player_vals,
            high=high_player_vals,
            shape=(self.PLAYER_CHANNELS,),
            dtype=np.float32
        )

        # Used for masking uninformed decisions
        self.NUM_ACTIONS = self.ROWS * self.COLS + 1
        mask_space = gym.spaces.Box(
            low=np.zeros((self.NUM_ACTIONS,), dtype=np.float32),
            high=np.ones((self.NUM_ACTIONS,), dtype=np.float32),
            shape=(self.NUM_ACTIONS,),
            dtype=np.float32
        )

        # Combine Board and Player and Mask representations into a single observation space
        self.observation_space = gym.spaces.Dict({
            "board": board_space,
            "player": player_space,
            "mask": mask_space
        })

        # Action space: 0-129 for selected grid cells, 130 for level-up
        self.action_space = gym.spaces.Discrete(self.ROWS * self.COLS + 1)

        # Preallocate buffers to avoid creating new arrays every step
        self._board_buffer = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32)
        self._player_buffer = np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32)
        self._mask_buffer = np.zeros((self.NUM_ACTIONS,), dtype=np.float32)


    def _get_obs(self):
        """
        Translate the environment state into an observation for the agent.

        Observation Structure:
        - Board: 7 x 10 x 13 tensor with channels [Adj. Power, Adj. Bombs, Cell Power, 4 one-hot types]
        - Player: [Current HP, Max HP, Current XP, XP Capacity]

        :return: Dictionary containing 'board' and 'player' observations
        """

        # Translate game to board space and mask space
        board_space = self._board_buffer
        board_space.fill(0.0)
        mask_space = self._mask_buffer
        mask_space.fill(1.0) # Start with all being legal

        # Loop through every cell and translate it
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # Adjacent Power, adjacent mines, and cell power are pretty much always unknown
                board_space[self.ADJ_POWER_IDX, row, col] = self.UNKNOWN_VALUE
                board_space[self.ADJ_BOMBS_IDX, row, col] = self.UNKNOWN_VALUE
                board_space[self.CELL_POWER_IDX, row, col] = self.UNKNOWN_VALUE

                # If cell is hidden, we have no information on it
                if not cell.revealed:
                    continue

                actor = cell.actor

                # If cell is empty, then no power, and we only know info if not obscured.
                if actor in [Actors.EMPTY, Actors.NONE]:
                    board_space[self.EMPTY_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = 0.0
                    if not cell.obscured:
                        adj_bombs = cell.adj_power // 100
                        adj_power = cell.adj_power % 100
                        board_space[self.ADJ_POWER_IDX, row, col] = min(1.0, adj_power / self.MAX_HP)
                        board_space[self.ADJ_BOMBS_IDX, row, col] = adj_bombs / self.MAX_ADJ_BOMBS

                # If cell is chest or mimic, we indicate it as such and set cell power to zero
                elif actor in [Actors.CHEST or Actors.MIMIC]:
                    board_space[self.CHEST_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = 0.0

                # If cell is medikit, we indicate it as such and set cell power to zero
                elif actor == Actors.MEDIKIT:
                    board_space[self.MEDIKIT_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = 0.0

                # If cell is a mine, we set cell power to 20 if it is not defused
                # If it is defused, we simply treat it as a safe cell and let the power be zero
                elif actor == Actors.MINE and cell.power != 0:
                    board_space[self.CELL_POWER_IDX, row, col] = 1.0

                # If cell is a wall, we simply indicate that it will deal 1 damage
                elif actor == Actors.WALL:
                    board_space[self.WALL_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = self.game.board[row][col].power / self.MAX_CELL_POWER

                # In any other case, we simply indicate the power of the cell
                else:
                    board_space[self.CELL_POWER_IDX, row, col] = min(1.0, cell.power / self.MAX_CELL_POWER)

        # Mask any illegal moves
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # If revealed, it is legal only if it is not empty
                if cell.revealed:
                    if cell.actor in [Actors.EMPTY, Actors.NONE]:
                        mask_space[row * self.COLS + col] = 0.0
                    continue

                # Check if at least 1 surrounding cell provides information on it
                adjacent_information = False
                for row_sum in [-1, 0, 1]:
                    if adjacent_information:
                        break

                    for col_sum in [-1, 0, 1]:
                        # Skip if it's the selected cell
                        if row_sum == col_sum == 0:
                            continue

                        # Get new row and col
                        new_row = row + row_sum
                        new_col = col + col_sum

                        # Skip if out of bounds
                        if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                            continue

                        if board_space[self.ADJ_POWER_IDX, new_row, new_col] >= 0.0 or board_space[self.ADJ_BOMBS_IDX, new_row, new_col] >= 0.0:
                            adjacent_information = True
                            break

                # If no adjacent information then mask it
                if not adjacent_information:
                    mask_space[row * self.COLS + col] = 0.0

        # Mask level up if not enough XP
        mask_space[-1] = 1.0 if self.game.xp >= self.game.get_required_level_xp() else 0.0

        # Translate game to player space
        player_space = self._player_buffer
        player_space[self.CURRENT_HP_IDX] = self.game.curr_health / self.MAX_HP
        player_space[self.MAX_HP_IDX] = self.game.max_health / self.MAX_HP
        player_space[self.CURRENT_XP_IDX] = min(1.0, self.game.xp / self.MAX_XP_REQUIRED)
        player_space[self.XP_REQUIRED_IDX] = min(1.0, self.game.get_required_level_xp() / self.MAX_XP_REQUIRED)

        # Return Observation
        return {"board": board_space, "player": player_space, "mask": mask_space}


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

        # Render if required
        if self.render_mode == "human" and self.game_visual:
            self.render()

        return self._get_obs(), self._get_info()


    # Convert an action to a board position *assuming it can be converted*
    # This function deliberately doesn't have a check for levelling up
    # Since such a guard would force everyone to include a potentially superfluous if
    def _action_pos(self, action: int):
        ROW = action // self.COLS
        COL = action % self.COLS
        return ROW, COL


    def _calculate_reward(self, old_board, action, win, alive, success, prev_hp):

        """
        Computes reward based on game state and action

        :return: The reward calculated
        """
        # Penalize nonsense moves heavily, this should never happen
        if not success:
            return self.REWARD_NONSENSE

        # If the agent dies, give large negative reward (but less than nonsense)
        if not alive:
            return self.REWARD_DEATH

        if win:  # Reward winning heavily (though this will likely enver occur)
            return self.REWARD_WIN

        # We need row and col for the following checks
        row, col = self._action_pos(action)

        # Reward based on how effective the level up was
        if action == self.LEVEL_UP_INDEX or old_board[self.MEDIKIT_IDX, row, col]:
            bonus = 0.1 if action == self.LEVEL_UP_INDEX else 0  # Prioritize levelling over healing
            if prev_hp == 1:
                return self.REWARD_PERFECT_LEVEL_OR_HEAL + bonus  # Perfect level up
            elif prev_hp == 2:
                return self.REWARD_PERFECT_LEVEL_OR_HEAL + bonus  # Slightly off
            else:
                return self.REWARD_BAD_LEVEL_OR_HEAL  # Inefficient

        # Reward clicking CHEST (or MIMIC and not dying)
        if old_board[self.CHEST_IDX, row, col]:
            return self.REWARD_CHEST_MIMIC

        # Clicking anything SAFE should be heavily rewarded (agent should always take this action is available)
        if not old_board[self.EMPTY_IDX, row, col] and old_board[self.CELL_POWER_IDX, row, col] == 0:
            return self.REWARD_SAFE_SQUARE


        # If the agent clicked a revealed cell that must have dealt damage
        if old_board[self.CELL_POWER_IDX, row, col] > self.UNKNOWN_VALUE:
            # Return a reward if health usage is maximized
            if self.game.curr_health == 1:
                return self.REWARD_OPTIMAL_HP_KILL
            # Otherwise, return a standard reward
            return self.REWARD_NON_OPTIMAL_HP_KILL

        # Here's the hard part. We want to reward SMART selection of unknown squares.
        # 1) if any of the  surrounding squares have a zero, we know it was safe to click, so all good.
        # 2) Check if the number surrounding it is lower than our current health, if it was good, otherwise risky
        # 3) If there's NO information around it, that is very bad (agent just guessed randomly)
        adj_revealed = False

        for row_sum in [-1, 0, 1]:
            for col_sum in [-1, 0, 1]:
                # Skip if it's the selected cell
                if row_sum == col_sum == 0:
                    continue

                # Get new row and col
                new_row = row + row_sum
                new_col = col + col_sum

                # Skip if out of bounds
                if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                    continue

                # Get cell information
                revealed = old_board[self.ADJ_POWER_IDX, new_row, new_col] > self.UNKNOWN_VALUE
                empty = old_board[self.EMPTY_IDX, new_row, new_col]
                adj_power = old_board[self.ADJ_POWER_IDX, new_row, new_col]
                adj_mines = old_board[self.ADJ_BOMBS_IDX, new_row, new_col]

                # If the cell isn't revealed and empty, then there's no information on it
                if not (revealed and empty):
                    continue

                adj_revealed = True

                # Return good reward if cell indicates that the cell clicked was safe to click (point 1)
                if adj_power == 0 and adj_mines == 0:  # Proves 1)
                    return self.REWARD_GUARANTEED_SAFE

                # Return decent reward if cell indicates that the cell clicked wouldn't kill us (point 2)
                if adj_power < prev_hp and adj_mines == 0:  # Prove 2)
                    return self.REWARD_GUARANTEED_NO_DEATH

        # Penalize if the agent just made a random guess with no information (point 3)
        if not adj_revealed:
            return self.REWARD_RANDOM_GUESS


        # If the agent explores and doesn't die.
        return self.REWARD_OTHER_ACTIONS


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

        # Get observation (IMPORTANT THAT THIS IS AFTER REWARD)
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