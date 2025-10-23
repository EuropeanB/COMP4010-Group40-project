import pygame, sys
from Game import Game
from Actors import Actors
from GameCell import Cell

class GameVisual:
    def __init__(self):
        pygame.init()

        # Set window size
        WINDOW_WIDTH = (64*13) + (8 * 12)
        WINDOW_HEIGHT = (64*10) + (8 * 9) + 88

        self.clock = pygame.time.Clock()

        # Create the window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("DragonSweeper")

        self.load_sprites()

        self.font = pygame.font.Font(None, 32)

    def set_game(self, game):
        self.game = game

    def load_sprites(self):
        self.sprites = {}
        self.sprites[Actors.BAT] = pygame.image.load("Sprites/bat.png").convert()
        self.sprites[Actors.DRAGON] = pygame.image.load("Sprites/dragon.png").convert()
        self.sprites[Actors.RAT] = pygame.image.load("Sprites/rat.png").convert()
        self.sprites[Actors.GARGOYLE] = pygame.image.load("Sprites/gargoyle.png").convert()
        self.sprites[Actors.SLIME] = pygame.image.load("Sprites/slime.png").convert()
        self.sprites[Actors.BIG_SLIME] = pygame.image.load("Sprites/big_slime.png").convert()
        self.sprites[Actors.MINOTAUR] = pygame.image.load("Sprites/minotaur.png").convert()
        self.sprites[Actors.MINE_KING] = pygame.image.load("Sprites/mine_king.png").convert()
        self.sprites[Actors.GIANT] = pygame.image.load("Sprites/giant.png").convert()
        self.sprites[Actors.GUARD] = pygame.image.load("Sprites/guard.png").convert()
        self.sprites[Actors.GAZER] = pygame.image.load("Sprites/gazer.png").convert()
        self.sprites[Actors.ORB] = pygame.image.load("Sprites/orb.png").convert()
        self.sprites[Actors.SPELL_MAKE_ORB] = pygame.image.load("Sprites/spell_make_orb.png").convert()
        self.sprites[Actors.MEDIKIT] = pygame.image.load("Sprites/medikit.png").convert()
        self.sprites[Actors.SKELETON] = pygame.image.load("Sprites/skeleton.png").convert()
        self.sprites[Actors.EMPTY] = pygame.image.load("Sprites/empty_tile.png").convert()
        self.sprites[Actors.GNOME] = pygame.image.load("Sprites/gnome.png").convert()
        self.sprites[Actors.SPELL_DISARM] = pygame.image.load("Sprites/spell_disarm.png").convert()
        self.sprites[Actors.SPELL_REVEAL_RATS] = pygame.image.load("Sprites/spell_reveal_rats.png").convert()
        self.sprites[Actors.SPELL_REVEAL_SLIMES] = pygame.image.load("Sprites/spell_reveal_slimes.png").convert()
        self.sprites[Actors.WIZARD] = pygame.image.load("Sprites/wizard.png").convert()
        self.sprites[Actors.CHEST] = pygame.image.load("Sprites/chest.png").convert()
        self.sprites[Actors.CROWN] = pygame.image.load("Sprites/crown.png").convert()
        self.sprites[Actors.MIMIC] = pygame.image.load("Sprites/chest.png").convert()
        self.sprites[Actors.DRAGON_EGG] = pygame.image.load("Sprites/dragon_egg.png").convert()
        self.sprites[Actors.MINE] = pygame.image.load("Sprites/mine.png").convert()
        self.sprites["wall_1"] = pygame.image.load("Sprites/wall_1.png").convert()
        self.sprites["wall_2"] = pygame.image.load("Sprites/wall_2.png").convert()
        self.sprites["wall_3"] = pygame.image.load("Sprites/wall_3.png").convert()
        self.sprites["hidden_tile"] = pygame.image.load("Sprites/hidden_tile.png").convert()
        self.sprites["empty_tile"] = pygame.image.load("Sprites/empty_tile.png").convert()
        self.sprites["jorge"] = pygame.image.load("Sprites/jorge.png").convert()
        self.sprites["jorge_level_up"] = pygame.image.load("Sprites/jorge_level_up.png").convert()
        self.sprites["jorge_zero_health"] = pygame.image.load("Sprites/jorge_zero_health.png").convert()

    def update_display(self):

        self.screen.fill((30, 30, 60))

        for i in range(self.game.ROWS):
            for j in range(self.game.COLS):
                cell = self.game.board[i][j]
                self.draw_cell(cell, j, i)

        curr_level = self.game.level - 1 if self.game.level < len(self.game.XP_REQUIREMENTS) else len(self.game.XP_REQUIREMENTS) - 1

        if self.game.xp >= self.game.XP_REQUIREMENTS[curr_level]:
            self.screen.blit(self.sprites["jorge_level_up"], (8,720))
        elif self.game.curr_health == 0:
            self.screen.blit(self.sprites["jorge_zero_health"], (8,720))
        else:
            self.screen.blit(self.sprites["jorge"], (8,720))

        player_info_string = f"{self.game.curr_health}/{self.game.max_health} HP  |  {self.game.xp}/{self.game.XP_REQUIREMENTS[curr_level]} XP  |  (Score: {self.game.score})"
        text_surface = self.font.render(player_info_string, True, (255, 255, 255))
        self.screen.blit(text_surface, text_surface.get_rect(center=(464,760)))

        pygame.display.flip()

        gv.clock.tick(60)

    def draw_cell(self, cell : Cell, x, y):

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

    def end_visualization(self):
        pygame.quit()
        sys.exit()


def get_clicked_tile_coords(click_pos):
    grid_coord = (click_pos[0] // 72, click_pos[1] // 72)
    return grid_coord[0], grid_coord[1]

def click_in_grid(pos):
    if pos[0] < 0 or pos[0] > 928:
        return False
    if pos[1] < 0 or pos[1] > 712:
        return False
    return True

def click_on_jorge(pos):
    if pos[0] > 8 and pos[0] < 80:
        if pos[1] > 720 and pos[1] < 800:
            return True
    return False

if __name__ == "__main__":
    gv = GameVisual()
    game = Game()
    game.reset_game()
    gv.set_game(game)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = event.pos  # (x, y)
                if click_in_grid(click_pos):
                    col, row = get_clicked_tile_coords(event.pos)
                    alive, terminated = game.touch_square(row, col)
                    if alive and terminated:
                        print(f"Game Over! Won with a score of {game.score}/365")
                        running = False
                    elif not alive:
                        print("Game Over! You Died!")
                        running = False
                if click_on_jorge(click_pos):
                    success = game.level_up()
                    if success:
                        print("You successfully levelled up! Press enter to continue")
                    else:
                        print("You do not have enough XP to level up. Press enter to continue")

        gv.update_display()
    
    gv.end_visualization
        
    



