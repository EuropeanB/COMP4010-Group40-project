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
        self.LEGAL_VALUE = 1.0
        self.ILLEGAL_VALUE = 0.0

        # Board space indices
        self.BOARD_CHANNELS = 10

        # Standard channels
        self.ADJ_POWER_IDX = 0
        self.CELL_POWER_IDX = 1
        self.ADJ_BOMBS_IDX = 2

        # One-hot encoded channels
        self.OBSCURED_IDX = 3
        self.EMPTY_IDX = 4
        self.SAFE_INDEX = 5
        self.MEDIKIT_IDX = 6
        self.CHEST_IDX = 7
        self.WALL_IDX = 8
        self.ENEMY_IDX = 9

        # Player space indices
        self.PLAYER_CHANNELS = 2

        # Standard channels
        self.HP_RATIO_IDX = 0
        self.XP_RATIO_IDX = 1

        # Board representation constants
        self.MAX_ADJ_POWER = 40 # This isn't trivially known. This is an estimate
        self.MAX_CELL_POWER = 20 # Highest damaging enemy is the dragon at 13, but we represent mines as 20 (instant kill, max hp)
        self.MAX_ADJ_BOMBS = 8

        # Board representation: Each index represents a square on the board
        # Channels: [Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [OBSCURED, EMPTY, SAFE, MEDIKIT, CHEST, WALL, ENEMY]
        # 5 channels + 6 one-hot encoding = 11 total channels
        board_space = gym.spaces.Box(
            low = np.zeros((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
            high = np.ones((self.BOARD_CHANNELS, self.ROWS, self.COLS), dtype=np.float32),
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
        - Board: 10 x 10 x 13 tensor with channels [Adj. Power, Adj. Bombs, Cell Power, 7 one-hot types]
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
                        board_space[self.ADJ_POWER_IDX, row, col] = min(1.0, adj_power / self.MAX_HP)
                        board_space[self.ADJ_BOMBS_IDX, row, col] = adj_bombs / self.MAX_ADJ_BOMBS

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
                    board_space[self.CELL_POWER_IDX, row, col] = self.game.board[row][col].power / self.MAX_CELL_POWER

                # In any other case it is an enemy, so we indicate it is a enemy and set its power
                else:
                    board_space[self.ENEMY_IDX, row, col] = 1.0
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

        # Mask level up if not enough XP
        mask_space[-1] = 1.0 if self.game.xp >= self.game.get_required_level_xp() else 0.0

        # Translate game to player space
        player_space = self._player_buffer
        player_space[self.HP_RATIO_IDX] = min(1.0, self.game.curr_health / self.MAX_HP)
        player_space[self.XP_RATIO_IDX] = min(1.0, self.game.xp / self.game.get_required_level_xp())

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


    def _calculate_reward(self, win, alive, success, level_up, actor_clicked, prev_hp, prev_max_hp, curr_hp, num_neighbours_revealed):
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
        # Nonsense move (masked, should never occur)
        if not success:
            return -20.0

        # Player dies
        if not alive:
            return -10.0

        # Player wins
        if win:
            return 10.0

        # Player levelled up or healed via medikitt
        if level_up or actor_clicked == Actors.MEDIKIT:
            bonus = 1.0 if level_up else 0.0
            if prev_hp == 1:
                return 3 + bonus
            elif prev_hp == 2:
                return 2 + bonus
            else:
                return -1

        # Always safe, always good
        if actor_clicked in self.SAFE_ACTORS:
            return 6.0

        # Clicked an unrevealed square
        if actor_clicked is None:
            # Blind guess
            if num_neighbours_revealed == 0:
                return -3.0

        # Calculated guess that didn't kill us
        return 1.0


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        prev_hp = self.game.curr_health # Health of the agent before update
        prev_max_hp = self.game.max_health # Max health of the agent before update
        actor_clicked = None # Actor clicked starts as unknown
        num_neighbours_revealed = 0 # Initialize to zero

        # Take action and check termination
        # Success is true if action did something, false otherwise
        if action == self.LEVEL_UP_INDEX:
            alive = True
            win = False
            level_up = True
            success = self.game.level_up()
        else:
            row, col = self._action_pos(action)
            num_neighbours_revealed = sum(
                1 if self.game.board[row][col].revealed and self.game.board[row][col].actor in [Actors.EMPTY, Actors.NONE]
                else 0
                for row, col in self.game.get_surrounding_cells((row, col), True)
            )
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
        reward = self._calculate_reward(win, alive, success, level_up, actor_clicked, prev_hp, prev_max_hp, curr_hp, num_neighbours_revealed)

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