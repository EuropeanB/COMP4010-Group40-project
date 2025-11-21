import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import minimum_filter, maximum_filter
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

        # Board space indices
        self.BOARD_CHANNELS = 5

        # Standard channels
        # Power Danger: Minimum damage we can expect to receive. This is normalized to the same scale as hp. So, 1 DMG is
        #    0.05, 2 DMG is 0.1, ..., to a cap of 20 DMG (MAX HP) which is 1.0 (or instant-death)
        # Mine Danger: 1.0 if cell has possibility of being a mine, 0.0 otherwise
        # Medikit: 1.0 if cell is a medikit, 0.0 otherwise
        # Wall: 1.0 if cell is a well, 0.0 otherwise
        # Clickable: 1.0 if cell is clickable (legal), 0.0 otherwise
        self.POWER_DANGER_IDX = 0 # Minimum damage we can expect to receive (normalized to the same scale as hp, so 1 dmg is 0.05,
        self.MINE_DANGER_IDX = 1 # 1.0 if cell has possibility of being a mine, 0.0 otherwise
        self.MEDIKIT_IDX = 2 # 1.0 if cell is a medikit, 0.0 otherwise
        self.WALL_IDX = 3 # 1.0 if cell is a wall, 0.0 otherwise
        self.CLICKABLE_IDX = 4 # 1.0 if cell is clickable (legal), 0.0 otherwise

        # Player space indices
        self.PLAYER_CHANNELS = 2

        # Standard channels
        # HP Ratio: Simply current HP normalized (/20.0)
        # XP Ratio: Progress to level up, 0.0 to 1.0
        self.HP_RATIO_IDX = 0
        self.XP_RATIO_IDX = 1

        # Board representation: Each index represents a square on the board
        # Channels: [Power Danger, Mine Danger, Medikit Flag, Wall Flag, Clickable Flag]
        board_space = gym.spaces.Box(
            low = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            high = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            shape=(self.BOARD_CHANNELS, self.ROWS, self.COLS),
            dtype=np.float32
        )

        # Player representation: [HP Ratio, XP Ratio]
        player_space = gym.spaces.Box(
            low=np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32),
            high=np.ones((self.PLAYER_CHANNELS,),  dtype=np.float32),
            shape=(self.PLAYER_CHANNELS,),
            dtype=np.float32
        )

        # Mask representation: Used simply for masking illegal moves (clicking empty revealed square, no XP level up)
        self.NUM_ACTIONS = self.ROWS * self.COLS + 1
        mask_space = gym.spaces.Box(
            low=np.zeros((self.NUM_ACTIONS,), dtype=np.float32),
            high=np.ones((self.NUM_ACTIONS,), dtype=np.float32),
            shape=(self.NUM_ACTIONS,),
            dtype=np.float32
        )

        # Combine Board, Player, and Mask representations into a single observation space
        self.observation_space = gym.spaces.Dict({
            "board": board_space,
            "player": player_space,
            "mask": mask_space
        })

        # Action space: 0-129 for selected grid cells, 130 for level-up
        self.action_space = gym.spaces.Discrete(self.ROWS * self.COLS + 1)

        # Required for reward function
        self.previous_obs = None

        # Preallocate buffers to avoid creating new arrays every step (for the reward function)
        self._board_buffer = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32)
        self._player_buffer = np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32)
        self._mask_buffer = np.zeros((self.NUM_ACTIONS,), dtype=np.float32)
        self.MAX_FLOAT = 1000.0 # Required for get obs
        self.revealed = np.zeros((self.ROWS, self.COLS),dtype=np.float32)  # Simply any revealed cell, 1.0 if revealed, 0.0 otherwise
        self.known_power = np.zeros((self.ROWS, self.COLS),dtype=np.float32)  # If enemy and revealed, mark its power. 0.0 otherwise
        self.known_surrounding_power = np.full((self.ROWS, self.COLS), self.MAX_FLOAT, dtype=np.float32)  # Power displayed by revealed and empty cells (NOT INCLUDING MINE POWER), MAX_INT otherwise
        self.not_touching_mine = np.zeros((self.ROWS, self.COLS),dtype=np.float32)  # 1.0 if confirmed not touching mine, 0.0 otherwise
        self.revealed_mine = np.zeros((self.ROWS, self.COLS),dtype=np.float32)  # 1.0 if it is revealed and a mine, 0.0 otherwise
        self.walls = np.zeros((self.ROWS, self.COLS), dtype=np.float32)  # If cell is a wall 1.0. 0.0 otherwise


    def _get_obs(self):
        # Reset buffer (safe)
        self._board_buffer.fill(0.0)
        self._board_buffer[self.CLICKABLE_IDX, :, :] = 1.0 # All moves start as clickable
        self._player_buffer.fill(0.0)
        self._mask_buffer.fill(1.0) # All moves start as legal
        self._mask_buffer[-1] = self.game.xp >= self.game.get_required_level_xp() # Mask level up if not enough xp

        # Reset tracking arrays (safe)
        self.revealed.fill(0.0) # Simply any revealed cell, 1.0 if revealed, 0.0 otherwise
        self.known_power.fill(0.0) # If enemy and revealed, mark its power. 0.0 otherwise
        self.known_surrounding_power.fill(self.MAX_FLOAT) # Power displayed by revealed and empty cells (NOT INCLUDING MINE POWER), MAX_INT otherwise
        self.not_touching_mine.fill(0.0) # 1.0 if confirmed not touching mine, 0.0 otherwise
        self.revealed_mine.fill(0.0) # 1.0 if it is revealed and a mine, 0.0 otherwise
        self.walls.fill(0.0) # If cell is a wall 1.0. 0.0 otherwise


        # Populate arrays using game logic
        for r in range(self.ROWS):
            for c in range(self.COLS):
                cell = self.game.board[r][c]

                if not cell.revealed:
                    continue

                self.revealed[r, c] = 1.0
                actor = cell.actor

                if actor in [Actors.EMPTY, Actors.NONE]:
                    self._board_buffer[self.CLICKABLE_IDX, r, c] = 0.0
                    self._mask_buffer[r * self.COLS + c] = 0.0
                    if not cell.obscured:
                        adj_bombs = cell.adj_power // 100
                        adj_power = cell.adj_power % 100
                        self.known_surrounding_power[r, c] = adj_power
                        self.not_touching_mine[r, c] = 1.0 if adj_bombs == 0 else 0.0

                # We treat it as unrevealed if it is a chest as we do not know if chest or mimic
                elif actor in [Actors.CHEST, Actors.MIMIC]:
                    self.revealed[r, c] = 0.0

                elif actor in self.SAFE_ACTORS:
                    continue # Do nothing for now

                elif actor == Actors.MEDIKIT:
                    self._board_buffer[self.MEDIKIT_IDX, r, c] = 1.0

                # Does not count towards power, only mines. Defused counts as safe
                elif actor == Actors.MINE:
                    if cell.power == 0.0: # Defused
                        continue
                    else:
                        self.revealed_mine[r, c] = 1.0

                elif actor == Actors.WALL:
                    self._board_buffer[self.WALL_IDX, r, c] = 1.0
                    self.walls[r, c] = 1.0

                # Actor is definitely an enemy
                else:
                    self.known_power[r, c] = cell.power

        # Calculate Power Danger
        # Step 1: Known surrounding power minus the sum of known power around cell (excludes self)
        kernel = np.ones((3, 3), dtype=np.float32)
        neighbour_sum_full = convolve2d(self.known_power, kernel, mode="same", boundary="fill", fillvalue=0)
        self.known_surrounding_power += self.known_power - neighbour_sum_full

        # Step 2: minimum value of surrounding cells from known surrounding power (mask w/ not revealed)
        min_possible_power = minimum_filter(self.known_surrounding_power, size=3, mode='constant', cval=self.MAX_FLOAT)
        min_possible_power = np.where(1.0 - self.revealed, min_possible_power, 0.0)

        # Step 3: add walls to min_possible_power to indicate that they will deal 1 damage
        min_possible_power += self.walls

        # Step 5: final output for power is: known power + min possible power (illegal move mask applied later)
        power_danger = self.known_power + min_possible_power
        power_danger = np.minimum(1.0, power_danger / self.POWER_NORMALIZER)
        self._board_buffer[self.POWER_DANGER_IDX, :, :] = power_danger

        # Calculate Mine danger
        # Step 1: Cells touching a cell flagged as "not touching a mine" are definitely not a mine
        not_a_mine = maximum_filter(self.not_touching_mine, size=3, mode="constant", cval=0)

        # Step 2: invert so it's cells that are potentially a mine, and mask w/ not revealed
        potential_mine = (1 - not_a_mine) * (1 - self.revealed)

        # Step 3: final output for mines is: potential mine + revealed mine
        self._board_buffer[self.MINE_DANGER_IDX, :, :] = (potential_mine + self.revealed_mine)

        # Translate game to player space
        self._player_buffer[self.HP_RATIO_IDX] = min(1.0, self.game.curr_health / self.HP_NORMALIZER)
        self._player_buffer[self.XP_RATIO_IDX] = min(1.0, self.game.xp / self.game.get_required_level_xp())

        # Return Observation
        return {"board": self._board_buffer, "player": self._player_buffer, "mask": self._mask_buffer}


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

        # Get previous observation, store internally for reward function
        self.previous_obs = self._get_obs()

        return self.previous_obs, self._get_info(None, False)


    def _action_pos(self, action):
        """
        Convert an action to a board position

        :param action: int representation of an action
        :return: Row, Col representation of the action
        """
        return action // self.COLS, action % self.COLS


    def _calculate_reward(self, win, alive, success, level_up, actor_clicked, prev_hp, prev_board, action):
        """
        Calculate the reward given for the resulting game state.

        :param win: True if game was won, False otherwise
        :param alive: True if player is alive, False otherwise
        :param success: False if player made an illegal move, True otherwise
        :param level_up: True if player levelled up, False otherwise
        :param actor_clicked: The actor the player clicked to lead to this state, None if click was unknown
        :param prev_hp: The player's previous HP
        :param prev_board: The board observation of the previous state
        :param action: The action the player took (0 - 130)
        :return: The total reward for the current game state
        """
        if not success:
            return -3.0

        # Death
        if not alive:
            return -10.0

        # Victory
        if win:
            return 20.0

        # Reward perfect healing, penalize bad healing
        if level_up or actor_clicked == Actors.MEDIKIT:
            if prev_hp == 1:
                return 1.5
            elif prev_hp == 2:
                return 0.3
            else:
                return -0.2

        row, col = self._action_pos(action)
        power_danger = prev_board[self.POWER_DANGER_IDX, row, col] * self.POWER_NORMALIZER
        mine_flag = prev_board[self.MINE_DANGER_IDX, row, col] == 1.0

        # Safe exploration should always be done first
        # This can also be accomplished with if in safe_actors or empty
        if power_danger == 0 and mine_flag == 0:
            return 0.3

        # Discourage bad exploration
        if power_danger == 1 or mine_flag == 1:
            return -1.0

        return 0.05


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        prev_hp = self.game.curr_health # Health of the agent before update
        actor_clicked = None # Actor clicked starts as unknown

        # Take action and check termination
        # Success is true if action was legal, false otherwise
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

        # Check termination
        terminated = not alive or win

        # Calculate reward
        reward = self._calculate_reward(win, alive, success, level_up, actor_clicked, prev_hp, self.previous_obs['board'], action)

        # Get observation
        observation = self._get_obs()

        # Store previous observation internally, for the reward calculation
        self.previous_obs = observation

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