import pygame, sys
from Game import Game
from Actors import Actors
from Cell import Cell

class GameVisual:
    """
    Handles visualization for the DragonSweeper game using Pygame.

    This class is responsible for:
    - Initializing the Pygame window
    - Loading sprites for all game actors
    - Rendering the board and player stats
    - Managing the game clock for frame rate control
    """
    def __init__(self, game):
        """
        Initialize the visualization environment.

        :param game: Instance of the Game class containing the board and game state
        """
        pygame.init()
        self.game = game

        # Set window size
        self.WINDOW_WIDTH = (64 * self.game.COLS) + (8 * 12)
        self.WINDOW_HEIGHT = (64 * self.game.ROWS) + (8 * 9) + 88

        # Create the window
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("DragonSweeper")
        self.clock = pygame.time.Clock()
        self.load_sprites()
        self.font = pygame.font.Font(None, 32)


    def load_sprites(self):
        """
        Loads all required sprite images into memory.
        Sprites are stored in a dictionary, keyed by actor type or name.
        """
        path = "Sprites"
        self.sprites = {}
        self.sprites[Actors.BAT] = pygame.image.load(f"{path}/bat.png").convert()
        self.sprites[Actors.DRAGON] = pygame.image.load(f"{path}/dragon.png").convert()
        self.sprites[Actors.RAT] = pygame.image.load(f"{path}/rat.png").convert()
        self.sprites[Actors.GARGOYLE] = pygame.image.load(f"{path}/gargoyle.png").convert()
        self.sprites[Actors.SLIME] = pygame.image.load(f"{path}/slime.png").convert()
        self.sprites[Actors.BIG_SLIME] = pygame.image.load(f"{path}/big_slime.png").convert()
        self.sprites[Actors.MINOTAUR] = pygame.image.load(f"{path}/minotaur.png").convert()
        self.sprites[Actors.MINE_KING] = pygame.image.load(f"{path}/mine_king.png").convert()
        self.sprites[Actors.GIANT] = pygame.image.load(f"{path}/giant.png").convert()
        self.sprites[Actors.GUARD] = pygame.image.load(f"{path}/guard.png").convert()
        self.sprites[Actors.GAZER] = pygame.image.load(f"{path}/gazer.png").convert()
        self.sprites[Actors.ORB] = pygame.image.load(f"{path}/orb.png").convert()
        self.sprites[Actors.SPELL_MAKE_ORB] = pygame.image.load(f"{path}/spell_make_orb.png").convert()
        self.sprites[Actors.MEDIKIT] = pygame.image.load(f"{path}/medikit.png").convert()
        self.sprites[Actors.SKELETON] = pygame.image.load(f"{path}/skeleton.png").convert()
        self.sprites[Actors.EMPTY] = pygame.image.load(f"{path}/empty_tile.png").convert()
        self.sprites[Actors.GNOME] = pygame.image.load(f"{path}/gnome.png").convert()
        self.sprites[Actors.SPELL_DISARM] = pygame.image.load(f"{path}/spell_disarm.png").convert()
        self.sprites[Actors.SPELL_REVEAL_RATS] = pygame.image.load(f"{path}/spell_reveal_rats.png").convert()
        self.sprites[Actors.SPELL_REVEAL_SLIMES] = pygame.image.load(f"{path}/spell_reveal_slimes.png").convert()
        self.sprites[Actors.WIZARD] = pygame.image.load(f"{path}/wizard.png").convert()
        self.sprites[Actors.CHEST] = pygame.image.load(f"{path}/chest.png").convert()
        self.sprites[Actors.CROWN] = pygame.image.load(f"{path}/crown.png").convert()
        self.sprites[Actors.MIMIC] = pygame.image.load(f"{path}/chest.png").convert()
        self.sprites[Actors.DRAGON_EGG] = pygame.image.load(f"{path}/dragon_egg.png").convert()
        self.sprites[Actors.MINE] = pygame.image.load(f"{path}/mine.png").convert()
        self.sprites["wall_1"] = pygame.image.load(f"{path}/wall_1.png").convert()
        self.sprites["wall_2"] = pygame.image.load(f"{path}/wall_2.png").convert()
        self.sprites["wall_3"] = pygame.image.load(f"{path}/wall_3.png").convert()
        self.sprites["hidden_tile"] = pygame.image.load(f"{path}/hidden_tile.png").convert()
        self.sprites["empty_tile"] = pygame.image.load(f"{path}/empty_tile.png").convert()
        self.sprites["jorge"] = pygame.image.load(f"{path}/jorge.png").convert()
        self.sprites["jorge_level_up"] = pygame.image.load(f"{path}/jorge_level_up.png").convert()
        self.sprites["jorge_zero_health"] = pygame.image.load(f"{path}/jorge_zero_health.png").convert()
        self.sprites["jorge_dead"] = pygame.image.load(f"{path}/jorge_dead.png").convert()


    def update_display(self):
        """
        Renders the entire game board, player sprite, and status text.

        Draws each cell, then overlays the player sprite and info text.
        Uses clock.tick(60) to limit frame rate to 60 FPS.
        """
        if not self.game:
            return

        self.screen.fill((30, 30, 60))
        for i in range(self.game.ROWS):
            for j in range(self.game.COLS):
                cell = self.game.board[i][j]
                self.draw_cell(cell, j, i)

        # Player sprite
        curr_level = self.game.level - 1 if self.game.level < len(self.game.XP_REQUIREMENTS) else len(self.game.XP_REQUIREMENTS) - 1
        if self.game.curr_health <= 0:
            self.screen.blit(self.sprites["jorge_dead"], (8,720))
        elif self.game.xp >= self.game.XP_REQUIREMENTS[curr_level]:
            self.screen.blit(self.sprites["jorge_level_up"], (8,720))
        elif self.game.curr_health == 0:
            self.screen.blit(self.sprites["jorge_zero_health"], (8,720))
        else:
            self.screen.blit(self.sprites["jorge"], (8,720))

        # Player info text
        player_info_string = f"{self.game.curr_health}/{self.game.max_health} HP  |  {self.game.xp}/{self.game.XP_REQUIREMENTS[curr_level]} XP  |  (Score: {self.game.score})"
        text_surface = self.font.render(player_info_string, True, (255, 255, 255))
        self.screen.blit(text_surface, text_surface.get_rect(center=(464,760)))

        pygame.display.flip()
        self.clock.tick(60)


    def draw_cell(self, cell : Cell, x, y):
        """
        Draws a single cell at the specified grid coordinates.

        :param cell: Cell object containing actor type, revealed state, and other properties
        :param x: Column index on the board
        :param y: Row index on the board
        """
        screen_coords = ((x*64) + (x*8), (y*64) + (y*8))
        text_surface_center = ((x*64) + (x*8)+32, (y*64) + (y*8)+32)

        if not cell.revealed:
            self.screen.blit(self.sprites["hidden_tile"], screen_coords)
        elif cell.actor == Actors.MIMIC:
            pass
        elif cell.actor == Actors.XP:
            text_surface = self.font.render(str(cell.xp), True, (255, 255, 0))
            self.screen.blit(self.sprites["empty_tile"], screen_coords)
            self.screen.blit(text_surface, text_surface.get_rect(center=text_surface_center))
        elif cell.actor == Actors.WALL:
            if cell.power == 3:
                self.screen.blit(self.sprites["wall_1"], screen_coords)
            elif cell.power == 2:
                self.screen.blit(self.sprites["wall_2"], screen_coords)
            elif cell.power == 1:
                self.screen.blit(self.sprites["wall_3"], screen_coords)
        elif cell.actor == Actors.EMPTY:
            if cell.obscured:
                text_surface = self.font.render("?", True, (255, 0, 255))
                self.screen.blit(self.sprites["empty_tile"], screen_coords)
                self.screen.blit(text_surface, text_surface.get_rect(center=text_surface_center))
            else:
                text_surface = self.font.render(str(cell.adj_power), True, (255, 255, 255))
                self.screen.blit(self.sprites["empty_tile"], screen_coords)
                self.screen.blit(text_surface, text_surface.get_rect(center=text_surface_center))
                return
        else:
            self.screen.blit(self.sprites[cell.actor], screen_coords)


    @staticmethod
    def close():
        """
        Close the environment and clean up resources.
        """
        pygame.quit()
