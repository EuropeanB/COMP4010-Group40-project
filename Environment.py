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

        # If a cell's value isn't known, it is set to this value:
        self.LEGAL_VALUE = 1.0
        self.ILLEGAL_VALUE = 0.0

        # Game Constants
        self.HP_NORMALIZER = 20
        self.MINE_NORMALIZER = 8
        self.POWER_NORMALIZER = self.HP_NORMALIZER # (To keep scaling the same)

        # Board space indices
        self.BOARD_CHANNELS = 4

        # Board Channels
        self.POWER_DANGER_IDX = 0 # (Danger level in regard to power, 1.0 max danger, 0.0 min danger)
        self.POSSIBLE_MINE_IDX = 1 # (1.0 if possibly a mine, 0.0 if definitely not a mine)
        self.SAFE_IDX = 2 # (1.0 if safe action, 0.0 otherwise)
        self.MEDIKIT_IDX = 3 # (1.0 if medikit, 0.0 otherwise)

        # Player space indices
        self.PLAYER_CHANNELS = 2

        # Standard channels
        self.HP_RATIO_IDX = 0
        self.XP_RATIO_IDX = 1

        # Board representation: Each index represents a square on the board
        # Channels: [Power Danger Level, Mine Danger Level, Safe Flag, Medikit Flag]
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


    def _new_get_obs(self):
        # Standard masks
        medikit_mask = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        safe_mask = np.zeros((self.ROWS, self.COLS), dtype=np.float32)

        # Used for mine flags computation
        revealed_mine_mask = np.full((self.ROWS, self.COLS), False, dtype=bool)
        revealed_mask = np.full((self.ROWS, self.COLS), False, dtype=bool)
        mine_indicator_mask = np.full((self.ROWS, self.COLS), False, dtype=bool)

        # Used for power danger computation
        revealed_and_non_enemy_mask = np.full((self.ROWS, self.COLS), False, dtype=bool) # True if revealed and non-enemy, False otherwise
        displayed_power_mask = np.ones((self.ROWS, self.COLS), dtype=np.float32) # 1.0 by default, the normalized displayed power if revealed and empty
        actual_power_mask = np.zeros((self.ROWS, self.COLS), dtype=np.float32) # 0.0 by default, the actual power of the cell (for ex. if an enemy)

        # BOARD SPACE AND MASK SPACE
        mask_space = np.ones((self.NUM_ACTIONS,), dtype=np.float32)
        mask_space[-1] = 1.0 if self.game.xp >= self.game.get_required_level_xp() else 0.0 # Mask level up if not enough XP

        # Initial pass over board. Gather information
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # If cell is not revealed, no info so skip. Treat chests as hidden, no info on it
                if (not cell.revealed) or (cell.actor in [Actors.CHEST, Actors.MIMIC]):
                    continue

                # Indicate the cell is revealed
                revealed_mask[row][col] = True

                # If cell is empty and not obscured, it provides an adjacent power indicator
                if cell.actor in [Actors.EMPTY, Actors.NONE]:
                    mask_space[row * self.COLS + col] = 0.0 # move is illegal
                    revealed_and_non_enemy_mask[row][col] = True
                    if cell.obscured:
                        continue
                    displayed_power_mask[row][col] = min(1.0, (cell.adj_power % 100) / self.POWER_NORMALIZER)
                    if cell.adj_power // 100 > 0:
                        mine_indicator_mask[row][col] = True

                # If cell is a medikit, simply indicate it
                elif cell.actor == Actors.MEDIKIT:
                    revealed_and_non_enemy_mask[row][col] = True
                    medikit_mask[row][col] = 1.0

                # If cell is a safe actor, simply indicate it
                elif cell.actor in self.SAFE_ACTORS:
                    revealed_and_non_enemy_mask[row][col] = True
                    safe_mask[row][col] = 1.0

                # If cell is a mine, treat as safe if defused, otherwise it is marked
                elif cell.actor == Actors.MINE:
                    if cell.power == 0:
                        safe_mask[row][col] = 1.0
                    else:
                        revealed_mine_mask[row][col] = True
                        actual_power_mask[row][col] = 1.0

                # If a wall, we treat it as a 1HP enemy (probably should change this)
                elif cell.actor == Actors.WALL:
                    actual_power_mask[row][col] = 1.0 / self.POWER_NORMALIZER

                # Standard Enemy
                else:
                    actual_power_mask[row][col] = min(1.0, (cell.power % 100) / self.POWER_NORMALIZER)

        # Calculate mine flags - To qualify, needs to:
        # 1) Have at least 1 surrounding cell with 100+ adjacent power AND be not revealed, or
        # 2) Be revealed and a mine
        padded_mine_indicator_mask = np.pad(mine_indicator_mask, 1, constant_values=0)
        windows =np.lib.stride_tricks.as_strided(
            padded_mine_indicator_mask,
            shape=(self.ROWS, self.COLS, 3, 3),
            strides=padded_mine_indicator_mask.strides + padded_mine_indicator_mask.strides
        )
        mine_flags_mask = windows.max(axis=(2,3))
        mine_flags_mask = (mine_flags_mask & ~revealed_mask) | revealed_mine_mask
        mine_flags_mask = mine_flags_mask.astype(np.float32)

        # Calculate Power Danger - Process:
        # 1) Get the minimum of all surrounding squares, for each cell
        # 2)
        padded_displayed_power_mask = np.pad(displayed_power_mask, 1, constant_values=1.0)
        windows = np.lib.stride_tricks.as_strided(
            padded_displayed_power_mask,
            shape=(self.ROWS, self.COLS, 3, 3),
            strides=padded_displayed_power_mask.strides + padded_displayed_power_mask.strides
        )
        min_possible_power_mask = windows.min(axis=(2,3))
        min_possible_power_mask = np.where(revealed_and_non_enemy_mask, 0.0, min_possible_power_mask)
        final_power_mask = np.where(actual_power_mask > 0, actual_power_mask, min_possible_power_mask)


        # Stack into final output
        board_space = np.stack([final_power_mask, mine_flags_mask, safe_mask, medikit_mask], axis=0).astype(np.float32)

        # PLAYER SPACE
        player_space = np.zeros((self.PLAYER_CHANNELS,), dtype=np.float32)
        player_space[self.HP_RATIO_IDX] = min(1.0, self.game.curr_health / self.HP_NORMALIZER)
        player_space[self.XP_RATIO_IDX] = min(1.0, self.game.xp / self.game.get_required_level_xp())

        # Return Observation
        return {"board": board_space, "player": player_space, "mask": mask_space}


    '''def _get_obs(self):
        """
        Translate the environment state into an observation for the agent.

        Observation Structure:
        - Board (CHW): 4 x 10 x 13 tensor with channels [Power Danger Level, Mine Danger Level, Safe Flag, Medikit Flag]
        - Player: [Hp Ratio, XP Ratio]

        :return: Dictionary containing 'board', 'mask' and 'player' observations
        """
        # BOARD SPACE
        board_space = self._board_buffer
        board_space.fill(0.0)

        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.game.board[row][col]

                # If cell is not revealed, we look to surrounding cells.
                # It's power danger level, is the minimum value of surrounding, revealed, cells
                # If there's a potential mine, we flag that
                # Additionally, we treat chest as unknown, as we rely on surrounding info to know if it is a mimic
                if (not cell.revealed) or (cell.revealed and cell.actor in [Actors.CHEST, Actors.MIMIC]):
                    neighbours = self.game.get_surrounding_cells((row, col), True)
                    mine_flag = False
                    min_power = 1000

                    for n_row, n_col in neighbours:
                        n_cell = self.game.board[n_row][n_col]
                        if not (n_cell.revealed and n_cell.actor in [Actors.EMPTY, Actors.NONE]) or n_cell.obscured:
                            continue

                        mine_flag = True if n_cell.adj_power // 100 > 0 else mine_flag
                        power = n_cell.adj_power % 100
                        min_power = min(power, min_power)

                    board_space[self.POWER_DANGER_IDX][row][col] = min(1.0, min_power / self.POWER_NORMALIZER)
                    board_space[self.POSSIBLE_MINE_IDX][row][col] = 1.0 if mine_flag else 0.0

                else:
                    # If cell is empty, nothing to do
                    if cell.actor in [Actors.EMPTY, Actors.NONE]:
                        continue

                    # If cell is a safe, actor we indicate it as such
                    elif cell.actor in self.SAFE_ACTORS:
                        board_space[self.SAFE_IDX][row][col] = 1.0

                    # If cell is a medikit, we indicate it as such
                    elif cell.actor == Actors.MEDIKIT:
                        board_space[self.MEDIKIT_IDX][row][col] = 1.0

                    # If cell is a mine, we need to check if it's defused. Otherwise, max danger
                    elif cell.actor == Actors.MINE:
                        if cell.power == 0:
                            board_space[self.SAFE_IDX][row][col] = 1.0
                        else:
                            board_space[self.POWER_DANGER_IDX][row][col] = 1.0
                            board_space[self.POSSIBLE_MINE_IDX][row][col] = 1.0

                    # If cell is a wall, we treat it as a 1 damage enemy.
                    elif cell.actor == Actors.WALL:
                        board_space[self.POWER_DANGER_IDX][row][col] = 1.0 / self.POWER_NORMALIZER

                    # Otherwise, we simply indicate its danger level
                    else:
                        board_space[self.POWER_DANGER_IDX][row][col] = min(cell.power / self.POWER_NORMALIZER, 1.0)

        # MASK SPACE
        mask_space = self._mask_buffer
        mask_space.fill(1.0) # Start with all being legal

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

        # PLAYER SPACE
        player_space = self._player_buffer
        player_space[self.HP_RATIO_IDX] = min(1.0, self.game.curr_health / self.HP_NORMALIZER)
        player_space[self.XP_RATIO_IDX] = min(1.0, self.game.xp / self.game.get_required_level_xp())

        # Return Observation
        return {"board": board_space, "player": player_space, "mask": mask_space}'''


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

        '''obs = self._get_obs()
        new_obs = self._new_get_obs()

        def print_board(board):
            output = ""
            for r in range(10):
                for c in range(13):
                    output += f"{round(float(board[r, c]), 2)} "
                output += '\n'
            output += '\n\n'
            print(output)

        if not np.array_equal(obs['player'], new_obs['player']):
            print("BOARD ERROR:")
            for i in range(4):
                if not np.array_equal(obs['board'][i], new_obs['board'][i]):
                    print(f"CHANNEL {i}")
                    print("OLD (CORRECT)")
                    print_board(obs['board'][i])
                    print("NEW")
                    print_board(new_obs['board'][i])
            input("Continue")'''

        return self._new_get_obs(), self._get_info(None, False)


    # Convert an action to a board position *assuming it can be converted*
    # This function deliberately doesn't have a check for levelling up
    # Since such a guard would force everyone to include a potentially superfluous if
    def _action_pos(self, action: int):
        ROW = action // self.COLS
        COL = action % self.COLS
        return ROW, COL


    def _new_calculate_reward(self, success, alive, win, level_up, actor_clicked, prev_hp):
        # Illegal move, should never occur
        if not success:
            return -1.0

        # Player dies
        if not alive:
            return -2.0

        # Player wins the game (very rare)
        if win:
            return 20.0

        # Reward perfect healing
        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 1:
            return 0.8 + (0.2 if level_up else 0.0)

        # Reward near perfect healing
        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 2:
            return 0.4 + (0.2 if level_up else 0.0)

        # Penalize suboptimal healing
        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp >= 3:
            return -0.2

        # Agent clicked a known to be safe actor (Orb, XP, Gnome, Scroll, etc.)
        if actor_clicked in self.SAFE_ACTORS:
            return 1.0

        # Otherwise, we simply explored
        return 0.0


    '''def _calculate_reward(self, win, alive, success, level_up, actor_clicked, prev_hp, prev_max_hp, curr_hp, num_neighbours_revealed):
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
            return -1.0

        # Player dies
        if not alive:
            return -1.0

        # Play wins the game (insanely rare)
        if win:
            return 2.0

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 1:
            return 0.8 + (0.2 if level_up else 0.0)

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp == 2:
            return 0.4 + (0.2 if level_up else 0.0)

        if (level_up or actor_clicked == Actors.MEDIKIT) and prev_hp >= 3:
            return -0.2

        # Agent clicked a known to be safe actor (Orb, XP, Gnome, Scroll, etc.)
        if actor_clicked in self.SAFE_ACTORS:
            return 1.0

        # Hidden move
        if actor_clicked is None:
            # Uninformed move (Discourage blind guessing)
            if num_neighbours_revealed == 0:
                return -0.1

            # Informed guess
            elif num_neighbours_revealed >= 1:
                return 0.1

        # Clicked a revealed chest/mimic
        if actor_clicked in [Actors.CHEST, Actors.MIMIC]:
            return 0.4

        # Any other scenario is combat that we won with revealed:
        return 0.05'''


    def step(self, action):
        """
        Executes one timestep of the environment

        :param action: Integer action (0-129 for grid cells, 130 for level-up)
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        prev_hp = self.game.curr_health # Health of the agent before update
        prev_max_hp = self.game.max_health # Max health of the agent before update
        actor_clicked = None # Actor clicked starts as unknown
        #num_neighbours_revealed = 0 # Initialize to zero

        # Take action and check termination
        # Success is true if action did something, false otherwise
        if action == self.LEVEL_UP_INDEX:
            alive = True
            win = False
            level_up = True
            success = self.game.level_up()
        else:
            row, col = self._action_pos(action)
            '''num_neighbours_revealed = sum(
                1 if self.game.board[row][col].revealed and self.game.board[row][col].actor in [Actors.EMPTY, Actors.NONE]
                else 0
                for row, col in self.game.get_surrounding_cells((row, col), True)
            )'''
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
        reward = self._new_calculate_reward(success, alive, win, level_up, actor_clicked, prev_hp)

        # Get observation (IMPORTANT THAT THIS IS AFTER REWARD)
        #observation = self._get_obs()
        #new_observation = self._new_get_obs()
        observation = self._new_get_obs()

        # Get Truncated
        truncated = False

        # Get Info
        info = self._get_info(prev_hp, level_up)

        # Update render if required
        if self.render_mode == "human" and self.game_visual:
            self.render()

        # TESTING
        '''def print_board(board):
            output = ""
            for r in range(10):
                for c in range(13):
                    output += f"{round(float(board[r, c]), 2)} "
                output += '\n'
            output += '\n\n'
            print(output)

        if not np.array_equal(observation['player'], new_observation['player']):
            print("BOARD ERROR:")
            for i in range(4):
                if not np.array_equal(observation['board'][i], new_observation['board'][i]):
                    print(f"CHANNEL {i}")
                    print("OLD (CORRECT)")
                    print_board(observation['board'][i])
                    print("NEW")
                    print_board(new_observation['board'][i])
            input("Continue")'''

        return observation, reward, terminated, truncated, info


    def render(self, delay=0.0):
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