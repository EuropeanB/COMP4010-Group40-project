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

        # Board space indices
        self.BOARD_CHANNELS = 14

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
        self.ENEMY_IDX = 10
        self.OBSCURED_IDX = 11
        self.CROWN_IDX = 12
        self.MINE_IDX = 13

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
        # Channels: [Revealed, Adjacent Power, Cell Power, Adjacent Bombs, One-hot encoding]
        # One-hot encoding: [UNKNOWN, EMPTY, SAFE, WALL, HEALING_SCROLL, CHEST, ENEMY, OBSCURED, CROWN, MINE]
        # 4 channels + 10 one-hot encoding = 14 total channels
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
        '''player_space = gym.spaces.Box(
            low = np.array([self.MIN_HP, self.STARTING_HP_SLOTS, self.MIN_XP, self.STARTING_XP_REQUIRED], dtype=np.int32),
            high = np.array([self.MAX_HP, self.MAX_HP, self.MAX_XP, self.MAX_XP_REQUIRED], dtype=np.int32),
            shape=(self.PLAYER_CHANNELS,),
            dtype=np.int32
        )'''

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
        board_space = np.zeros((self.ROWS, self.COLS, self.BOARD_CHANNELS), dtype=np.float32)

        # Loop through every cell and translate it
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # Mark if cell is revealed
                board_space[row, col, self.REVEALED_IDX] = cell.revealed

                # If revealed and not obscured, mark adj power, cell power, and adjacent bombs
                if cell.revealed and not cell.obscured:
                    adj_bombs = cell.adj_power // 100
                    adj_power = cell.adj_power % 100

                    board_space[row, col, self.ADJ_POWER_IDX] = adj_power / self.MAX_ADJ_POWER
                    board_space[row, col, self.CELL_POWER_IDX] = cell.power / self.MAX_CELL_POWER
                    board_space[row, col, self.ADJ_BOMBS_IDX] = adj_bombs / self.MAX_ADJ_BOMBS

                # If not revealed or obscured, then we have no knowledge and set everything to zero
                else:
                    board_space[row, col, self.ADJ_POWER_IDX] = 0
                    board_space[row, col, self.CELL_POWER_IDX] = 0
                    board_space[row, col, self.ADJ_BOMBS_IDX] = 0

                # One-Hot Encode Type
                # Note that the order is important. For example, if XP and obscured, we prioritize showing XP
                actor = cell.actor
                if not cell.revealed:
                    board_space[row, col, self.UNKNOWN_IDX] = 1
                elif actor in [Actors.EMPTY, Actors.NONE]:
                    if cell.obscured: board_space[row, col, self.OBSCURED_IDX] = 1
                    else: board_space[row, col, self.EMPTY_IDX] = 1
                elif actor == Actors.WALL:
                    board_space[row, col, self.WALL_IDX] = 1
                elif actor in [Actors.CHEST, Actors.MIMIC]:
                    board_space[row, col, self.CHEST_IDX] = 1
                elif actor == Actors.MEDIKIT:
                    board_space[row, col, self.HEALING_SCROLL_IDX] = 1
                elif actor in [
                    Actors.ORB, Actors.SPELL_MAKE_ORB, Actors.SPELL_DISARM, Actors.SPELL_REVEAL_RATS,
                    Actors.SPELL_REVEAL_SLIMES, Actors.DRAGON_EGG, Actors.XP, Actors.GNOME
                ]:
                    board_space[row, col, self.SAFE_IDX] = 1
                elif actor == Actors.CROWN:
                    board_space[row, col, self.CROWN_IDX] = 1
                elif actor == Actors.MINE:
                    board_space[row, col, self.MINE_IDX] = 1
                else:
                    board_space[row, col, self.ENEMY_IDX] = 1

        # Translate game to player space
        player_space = np.array([
            self.game.curr_health / self.MAX_HP,
            self.game.max_health / self.MAX_HP,
            self.game.xp / self.MAX_XP,
            self.game.get_required_level_xp() / self.MAX_XP_REQUIRED
        ])

        # Return Observation
        return {
            "board": board_space,
            "player": player_space
        }


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
            "last touched": self.game.last_touched,
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
        self.game.reset_game()

        return self._get_obs(), self._get_info()
    
    # Convert an action to a board position *assuming it can be converted* 
    # This function deliberately doesn't have a check for levelling up
    # Since such a guard would force everyone to include a potentially superfluous if
    def _action_pos(self, action: int):
        ROW = action // self.COLS
        COL = action % self.COLS
        return (ROW, COL)

    def _calculate_reward(self, old_obs, action: int, new_obs, win: bool, alive: bool):
        '''
        Computes reward based on game state and action

        :return: The reward calculated
        '''
        # If the player has won, the reward is just their score
        if win:
            return float(self.game.score)

        # If the player is dead, then they receive a negative penalty
        if not alive:
            return -5.

        # Reward for any case where the player tries an action that does nothing
        NONSENSE = -1

        # "o" as in old
        o_board = old_obs['board']
        o_player = old_obs['player']

        # "n" as in new
        n_board = new_obs['board']
        n_player = new_obs['player']

        # If the player tried to level up
        if action == self.LEVEL_UP_INDEX:
            # If they can't level up
            if o_player[self.CURRENT_XP_IDX] < o_player[self.XP_REQUIRED_IDX]:
                return NONSENSE# Punish them for trying a move that does nothing
            # Otherwise, reward them for succesfully levelling up
            return 5.

        ROW, COL = self._action_pos(action)
        # If the player clicked a square where they already can see something about the square
        # And it's also empty...
        if o_board[ROW, COL, self.REVEALED_IDX] and o_board[ROW, COL, self.EMPTY_IDX]:
            return NONSENSE # Punish them for trying a move that does nothing
    
        # d short for delta, changes in game state
        dxp = n_player[self.CURRENT_XP_IDX] - o_player[self.CURRENT_XP_IDX] # dExperience
        drev = np.sum(n_board[:, :, self.REVEALED_IDX]) - np.sum(o_board[:, :, self.REVEALED_IDX]) # Number of newly revealed

        # If we're not in any of the other special cases
        # Then reward is just increase in exp times the value of each exp point
        # Plus the increase in revealed board positions times the value of revealing
        return (3 * dxp) + (2 * drev)

    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        old_obs = self._get_obs()

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

        observation = new_obs = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward(old_obs, action, new_obs, win, alive)

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