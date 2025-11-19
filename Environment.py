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
        self.SAFE_ACTORS = [Actors.ORB, Actors.SPELL_MAKE_ORB, Actors.SPELL_DISARM, Actors.SPELL_REVEAL_RATS,Actors.SPELL_REVEAL_SLIMES, Actors.DRAGON_EGG, Actors.XP, Actors.GNOME, Actors.CROWN]

        # Game Constants
        self.HP_NORMALIZER = 20
        self.POWER_NORMALIZER = 20
        self.MINE_NORMALIZER = 8

        # If a cell's value isn't known, it is set to this value:
        self.LEGAL_VALUE = 1.0
        self.ILLEGAL_VALUE = 0.0

        # Board space indices
        self.BOARD_CHANNELS = 11

        # Standard channels
        self.ADJ_POWER_IDX = 0
        self.CELL_POWER_IDX = 1
        self.ADJ_BOMBS_IDX = 2

        # One-hot encoded channels
        self.HIDDEN_IDX = 3
        self.OBSCURED_IDX = 4
        self.EMPTY_IDX = 5
        self.SAFE_INDEX = 6
        self.MEDIKIT_IDX = 7
        self.CHEST_IDX = 8
        self.WALL_IDX = 9
        self.ENEMY_IDX = 10

        # Player space indices
        self.PLAYER_CHANNELS = 2

        # Standard channels
        self.HP_RATIO_IDX = 0
        self.XP_RATIO_IDX = 1

        # Board representation: Each index represents a square on the board
        # Channels: [Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [HIDDEN, OBSCURED, EMPTY, SAFE, MEDIKIT, CHEST, WALL, ENEMY]
        # 5 channels + 6 one-hot encoding = 11 total channels
        board_space = gym.spaces.Box(
            low = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            high = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            shape=(self.BOARD_CHANNELS, self.ROWS, self.COLS),
            dtype=np.float32
        )

        # Player representation: [HP Ratio, XP Ratio]
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
        - Board: 11 x 10 x 13 tensor with channels [Adj. Power, Adj. Bombs, Cell Power, 8 one-hot types]
        - Player: [Hp Ratio, XP Ratio]

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

                # If cell is hidden, we have no information on it
                if not cell.revealed:
                    board_space[self.HIDDEN_IDX, row, col] = 1.0
                    continue

                actor = cell.actor

                # If cell is empty, then no power, and we only know info if not obscured.
                if actor in [Actors.EMPTY, Actors.NONE]:
                    if cell.obscured:
                        board_space[self.OBSCURED_IDX, row, col] = 1.0
                    else:
                        board_space[self.EMPTY_IDX, row, col] = 1.0
                        adj_bombs = cell.adj_power // 100
                        adj_power = cell.adj_power % 100
                        board_space[self.ADJ_POWER_IDX, row, col] = min(1.0, adj_power / self.POWER_NORMALIZER)
                        board_space[self.ADJ_BOMBS_IDX, row, col] = adj_bombs / self.MINE_NORMALIZER

                # If cell is a safe actor, we indicate it as such
                elif actor in self.SAFE_ACTORS:
                    board_space[self.SAFE_INDEX, row, col] = 1.0

                # If cell is chest or mimic, we indicate it as such
                elif actor in [Actors.CHEST, Actors.MIMIC]:
                    board_space[self.CHEST_IDX, row, col] = 1.0

                # If cell is medikit, we indicate it as such
                elif actor == Actors.MEDIKIT:
                    board_space[self.MEDIKIT_IDX, row, col] = 1.0

                # If cell is a mine, we set cell power to 20 if it is not defused
                # If it is defused, we simply treat it as a safe cell and let the power be zero
                elif actor == Actors.MINE:
                    if cell.power > 0:
                        board_space[self.ENEMY_IDX, row, col] = 1.0
                        board_space[self.CELL_POWER_IDX, row, col] = 1.0
                    else:
                        board_space[self.SAFE_INDEX, row, col] = 1.0

                # If cell is a wall, we simply indicate that it will deal 1 damage
                elif actor == Actors.WALL:
                    board_space[self.WALL_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = self.game.board[row][col].power / self.POWER_NORMALIZER

                # In any other case it is an enemy, so we indicate it is a enemy and set its power
                else:
                    board_space[self.ENEMY_IDX, row, col] = 1.0
                    board_space[self.CELL_POWER_IDX, row, col] = min(1.0, cell.power / self.POWER_NORMALIZER)

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

                        if board_space[self.EMPTY_IDX, new_row, new_col] == 1.0:
                            adjacent_information = True
                            break

                # If no adjacent information then mask it
                if not adjacent_information:
                    mask_space[row * self.COLS + col] = 0.0

        # Mask level up if not enough XP
        mask_space[-1] = 1.0 if self.game.xp >= self.game.get_required_level_xp() else 0.0

        # Translate game to player space
        player_space = self._player_buffer
        player_space[self.HP_RATIO_IDX] = min(1.0, self.game.curr_health / self.HP_NORMALIZER)
        player_space[self.XP_RATIO_IDX] = min(1.0, self.game.xp / self.game.get_required_level_xp())

        # Return Observation
        return {"board": board_space, "player": player_space, "mask": mask_space}


    def _get_info(self, prev_hp, levelled_up):
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
            "level": self.game.level,
            "prev hp": prev_hp,
            'levelled up': levelled_up
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

        return self._get_obs(), self._get_info(None, False)


    # Convert an action to a board position *assuming it can be converted*
    # This function deliberately doesn't have a check for levelling up
    # Since such a guard would force everyone to include a potentially superfluous if
    def _action_pos(self, action: int):
        ROW = action // self.COLS
        COL = action % self.COLS
        return ROW, COL


    def _calculate_reward(self, win, alive, success, level_up, actor_clicked, prev_hp, action):
        """
        Computes reward based on game state and action.

        :param win: True if game was won, False otherwise
        :param alive: True if player remains alive, False otherwise
        :param success: True if move did something, False if nonsense
        :param level_up: True if the player levelled up, False otherwise
        :param actor_clicked: The actor the player clicked, None if cell was not revealed
        :param prev_hp: The previous HP of the player
        :param prev_max_hp: The previous Max HP of the player
        :param curr_hp: The current HP of the player
        :param num_neighbours_revealed: The number of neighbouring cells that provide information
        :return: the total reward for the step
        """
        # Illegal move (should never occur)
        if not success:
            return -3.0

        # Player dies
        if not alive:
            return -3.0

        # Play wins the game (insanely rare)
        if win:
            return 10.0

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 1:
            return 1.3 + (0.2 if level_up else 0.0)

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 2:
            return 0.6 + (0.2 if level_up else 0.0)

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp >= 3:
            return -0.5

        # Agent clicked a known to be safe actor (Orb, XP, Gnome, Scroll, etc.)
        if actor_clicked in self.SAFE_ACTORS:
            return 1.0

        # We don't like clicking walls. Small bonus if its safe and we don't die, but don't
        # want to incentivize clicking it
        if actor_clicked == Actors.WALL:
            return 0.05

        # Treat chest as unknown. If it is a risky click, don't do it because it could be a mimic
        if actor_clicked in [Actors.CHEST, Actors.MIMIC]:
            actor_clicked = None

        # Willingly entered combat and survived
        if actor_clicked is not None:
            return 0.3

        # Agent clicked an unknown square: reward based on risk levels
        # Since clicking an unrevealed cell doesn't modify the surrounding cells
        # until XP is picked up (which is covered by SAFE ACTORS), we can use the current game
        row = action // self.COLS
        col = action % self.COLS
        would_not_kill = False
        if actor_clicked is None:
            for row_sum in [-1, 0, 1]:
                for col_sum in [-1, 0, 1]:
                    # Skip if same cell
                    if row_sum == col_sum == 0:
                        continue

                    # Get new row and col
                    new_row = row + row_sum
                    new_col = col + col_sum

                    # Out of bounds
                    if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
                        continue

                    new_cell = self.game.board[new_row][new_col]
                    if not (new_cell.revealed and not new_cell.obscured and new_cell.actor in [Actors.EMPTY, Actors.NONE]):
                        continue

                    power = new_cell.adj_power % 100
                    mines = new_cell.adj_power // 100

                    if power == 0 and mines == 0:
                        return 1.0 # Guaranteed safe exploration
                    elif power < prev_hp and mines == 0:
                        would_not_kill = True

        if would_not_kill:
            return 0.3 # Safe exploration! Wasn't going to die

        # Otherwise, risky exploration!
        return -0.6


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        prev_hp = self.game.curr_health # Health of the agent before update
        actor_clicked = None # Actor clicked starts as unknown

        # Take action and check termination
        # Success is true if action did something, false otherwise
        if action == self.LEVEL_UP_INDEX:
            alive = True
            win = False
            level_up = True
            success = self.game.level_up()
        else:
            row, col = self._action_pos(action)
            if self.game.board[row][col].revealed: # Agent knew what the actor was
                actor_clicked = self.game.board[row][col].actor
            alive, win, success = self.game.touch_square(row, col)
            level_up = False

        if not success:
            print("NOT SUCCESS BUG WHATTATOW")

        # Get HP after update
        curr_hp = self.game.curr_health

        # Check termination
        terminated = not alive or win

        # Calculate reward
        reward = self._calculate_reward(win, alive, success, level_up, actor_clicked, prev_hp, action)

        # Get observation (IMPORTANT THAT THIS IS AFTER REWARD)
        observation = self._get_obs()

        # Get Truncated
        truncated = False

        # Get Info
        info = self._get_info(prev_hp, level_up)

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