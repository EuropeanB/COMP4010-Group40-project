import numpy as np
import gymnasium as gym
import math
from Game import Game
from Actors import Actors


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
        self.ROWS = 10
        self.COLS = 13
        self.LEVEL_UP_INDEX = self.ROWS * self.COLS # The action value for levelling up

        # Internal State
        self.game = Game()

        # Game Constants
        self.NUM_BOMBS = 9 # Used for normalizing the bomb value
        self.START_HP_SLOTS = 5
        self.MIN_HP = -1 # Represent all death states as -1 HP
        self.MAX_HP = 19
        self.START_XP_SLOTS = 5
        self.MIN_XP = 0
        self.MAX_XP = 50 # Because of XP overflow, this number isn't trivially known

        # Board space indices
        self.BOARD_CHANNELS = 15 # Total number of channels

        # Standard channels
        self.REVEALED_IDX = 0
        self.ADJ_POWER_IDX = 1
        self.CELL_POWER_IDX = 2
        self.ADJ_BOMBS_IDX = 3

        # One-hot encoded channels
        self.UNKNOWN_IDX = 4
        self.EMPTY_IDX = 5
        self.SAFE_IDX = 6
        self.WALL_IDX = 7
        self.HEALING_SCROLL_IDX = 8
        self.CHEST_IDX = 9
        self.DRAGON_IDX = 10
        self.ENEMY_IDX = 11
        self.OBSCURED_IDX = 12
        self.CROWN_IDX = 13
        self.MINE_IDX = 14

        # Player space indices
        self.PLAYER_CHANNELS = 4

        # Standard channels
        self.CURRENT_HP_IDX = 0
        self.MAX_HP_IDX = 1
        self.CURRENT_XP_IDX = 2
        self.XP_REQUIRED_IDX = 3

        # Board representation: Each index represents a square on the board
        # Channels: [Revealed, Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [UNKNOWN, EMPTY, SAFE, WALL, HEALING_SCROLL, CHEST, DRAGON, ENEMY, OBSCURED]
        # 4 channels + 9 one-hot encoding = 13 total channels
        board_space = gym.spaces.Box(
            low = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            high = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            shape=(self.BOARD_CHANNELS, self.ROWS, self.COLS),
            dtype=np.float32
        )

        # Player representation: [Current HP, HP Slots, Current XP, XP Required to Level Up]
        player_space = gym.spaces.Box(
            low = np.array([self.MIN_HP, self.START_HP_SLOTS, self.MIN_XP, self.START_XP_SLOTS], dtype=np.int32),
            high = np.array([self.MAX_HP, self.MAX_HP, self.MAX_XP, self.MAX_XP], dtype=np.int32),
            shape=(self.PLAYER_CHANNELS,),
            dtype=np.int32
        )

        # Combine Board and Player representations into a single observation space
        self.observation_space = gym.spaces.Dict({
            "board": board_space,
            "player": player_space
        })

        # Action space: 0-129 for selected grid cells, 130 for level-up
        self.action_space = gym.spaces.Discrete(self.ROWS * self.COLS + 1)


    def _get_obs(self):
        """
        Translate the environment state into an observation for the agent.

        Observation Structure:
        - Board: 13 x 10 x 12 tensor with channels [Revealed, Adj. Power, Cell Power, Adj. Bombs, 9x one-hot types]
        - Player: [Current HP, Max HP, Current XP, XP Capacity]

        :return: Dictionary containing 'board' and 'player' observations
        """

        # Translate game to board space
        board_space = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32)

        # Loop through every cell and translate it
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # Mark if cell is revealed
                board_space[0, row, col] = float(cell.revealed)

                # If revealed and not obscured, mark adj power, cell power, and adjacent bombs
                if cell.revealed and not cell.obscured:
                    adj_bombs = math.floor(cell.adj_power / 100)
                    adj_power = cell.adj_power % 100

                    board_space[self.ADJ_POWER_IDX, row, col] = adj_power # Leaving this un-normalized for now, until we decide
                    board_space[self.CELL_POWER_IDX, row, col] = cell.power # Leaving this un-normalized for now, until we decide
                    board_space[self.ADJ_BOMBS_IDX, row, col] = adj_bombs / self.NUM_BOMBS

                # If not revealed or obscured, then we have no knowledge and set everything to zero
                else:
                    board_space[self.ADJ_POWER_IDX, row, col] = 0
                    board_space[self.CELL_POWER_IDX, row, col] = 0
                    board_space[self.ADJ_BOMBS_IDX, row, col] = 0

                # One-Hot Encode Type
                # Note that the order is important. For example, if XP and obscured, we prioritize showing XP
                actor = cell.actor
                if not cell.revealed:
                    board_space[self.UNKNOWN_IDX, row, col] = 1
                elif actor in [Actors.EMPTY, Actors.NONE]:
                    if cell.obscured: board_space[self.OBSCURED_IDX, row, col] = 1
                    else: board_space[self.EMPTY_IDX, row, col] = 1
                elif actor == Actors.DRAGON:
                    board_space[self.DRAGON_IDX, row, col] = 1
                elif actor == Actors.WALL:
                    board_space[self.WALL_IDX, row, col] = 1
                elif actor in [Actors.CHEST, Actors.MIMIC]:
                    board_space[self.CHEST_IDX, row, col] = 1
                elif actor == Actors.MEDIKIT:
                    board_space[self.HEALING_SCROLL_IDX, row, col] = 1
                elif actor in [
                    Actors.ORB, Actors.SPELL_MAKE_ORB, Actors.SPELL_DISARM, Actors.SPELL_REVEAL_RATS,
                    Actors.SPELL_REVEAL_SLIMES, Actors.DRAGON_EGG, Actors.XP, Actors.GNOME
                ]:
                    board_space[self.SAFE_IDX, row, col] = 1
                elif actor == Actors.CROWN:
                    board_space[self.CROWN_IDX, row, col] = 1
                elif actor == Actors.MINE:
                    board_space[self.MINE_IDX, row, col] = 1
                else:
                    board_space[self.ENEMY_IDX, row, col] = 1

        # Translate game to player space
        player_space = np.array([
            self.game.curr_health, self.game.max_health, self.game.xp, self.game.get_required_level_xp()
        ])

        # Return Observation
        return {
            "board": board_space,
            "player": player_space
        }


    def _get_info(self):
        """
        Returns diagnostic information for debugging/monitoring. Can implement later if needed

        :return: Auxiliary information associated to the current state
        """
        return {"info": None}


    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        :param seed: Random seed for reproducibility
        :param options: Additional options for environment reset
        :return: Tuple of (observation, info) for the initial state
        """
        super().reset(seed=seed)

        # Reset game
        self.game.reset_game()

        return self._get_obs(), self._get_info()


    def _calculate_reward(self):
        """
        Computes reward based on game state and action

        :return: The reward calculated
        """
        return 0


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        # Take action and check termination
        # Success is true if action did something, false otherwise
        if action == self.LEVEL_UP_INDEX:
            alive = True
            terminated = False
            success = self.game.level_up()
        else:
            row = math.floor(action / 13)
            col = action % 13
            alive, terminated, success = self.game.touch_square(row, col)

        # Calculate reward
        reward = self._calculate_reward()

        # Get observation
        observation = self._get_obs()

        # I'm not 100% sure what these two are used for, but they need to be returned
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info
