from Cell import Cell
import numpy as np
import gymnasium as gym


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
        self.COLS = 12
        self.BOARD_CHANNELS = 13
        self.PLAYER_CHANNELS = 4

        # Game Constants
        self.NUM_BOMBS = 9 # Used for normalizing the bomb value
        self.START_HP_SLOTS = 5
        self.MIN_HP = -1 # Represent all death states as -1 HP
        self.MAX_HP = 15 # I believe the maximum HP is 15 - needs confirmation
        self.START_XP_SLOTS = 5
        self.MIN_XP = 0
        self.MAX_XP = 25 # I believe the maximum XP is 25 - needs confirmation

        # Initialize state variables
        self.board = [[Cell() for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.player_hp = self.START_HP_SLOTS
        self.player_hp_slots = self.START_HP_SLOTS
        self.player_xp = 0
        self.player_xp_slots = self.START_XP_SLOTS

        # Board representation: Each index represents a square on the board
        # Channels: [Revealed, Adjacent Power, Cell Power, Adjacent Bombs, One-hot CellType encoding]
        board_space = gym.spaces.Box(
            low = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            high = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            shape=(self.BOARD_CHANNELS, self.ROWS, self.COLS),
            dtype=np.float32
        )

        # Player representation: [Current HP, HP Slots, Current XP, XP Slots]
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

        # Action space: 0-119 for selected grid cells, 120 for level-up
        self.action_space = gym.spaces.Discrete(self.ROWS * self.COLS + 1)


    def _get_obs(self):
        """
        Translate the environment state into an observation for the agent.

        Observation Structure:
        - Board: 13 x 10 x 12 tensor with channels [Revealed, Adj. Power, Cell Power, Adj. Bombs, 9x one-hot types]
        - Player: [Current HP, Max HP, Current XP, XP Capacity]

        :return: Dictionary containing 'board' and 'player' observations
        """

        # Translate board to board space
        board_space = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.board[row][col]

                # Mark if cell is revealed
                board_space[0, row, col] = float(cell.revealed)

                # If space is revealed, indicate surrounding power and bombs (normalized)
                if cell.revealed:
                    board_space[1, row, col] = cell.adj_power # Leaving this un-normalized for now, until we decide
                    board_space[2, row, col] = cell.cell_power  # Leaving this un-normalized for now, until we decide
                    board_space[3, row, col] = cell.adj_bombs / self.NUM_BOMBS

                # One-Hot Encode Type
                one_hot = cell.get_one_hot()
                board_space[4:12, row, col] = one_hot

        # Translate player to player space
        player_space = np.array([
            self.player_hp, self.player_hp_slots, self.player_xp, self.player_xp_slots
        ], dtype=np.int32)

        # Return observation
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

        # Reset board
        for row in self.board:
            for cell in row:
                cell.reset()

        # Reset player
        self.player_hp = self.START_HP_SLOTS
        self.player_hp_slots = self.START_HP_SLOTS
        self.player_xp = 0
        self.player_xp_slots = self.START_XP_SLOTS

        return self._get_obs(), self._get_info()


    def _calculate_reward(self):
        """
        Computes reward based on game state and action

        :return: The reward calculated
        """
        return 0


    def _check_termination(self):
        """
        Check if the game state is terminated based on the current state

        :return: True if terminated, false otherwise
        """
        return False


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-119 for grid cells, 120 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        # Take action
        # Here, ideally we have the game logic in python

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_termination()

        # Get observation
        observation = self._get_obs()

        # I'm not 100% sure what these two are used for, but they need to be return
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info
