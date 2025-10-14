from enum import Enum

class Actors(Enum):
    """
    This is all the possible actors that can be stored in a cell in the game
    """
    NONE = 0
    EMPTY = 1
    ORB = 2
    SPELL_MAKE_ORB = 3
    MINE = 4
    MINE_KING = 5
    DRAGON = 6
    WALL = 7
    MIMIC = 8
    MEDIKIT = 9
    RAT_KING = 10
    RAT = 11
    SLIME = 12
    GARGOYLE = 13
    MINOTAUR = 14
    CHEST = 15
    SKELETON = 16
    TREASURE = 17
    SNAKE = 18 # Pointless
    GIANT = 19
    WIZARD = 20
    GAZER = 21
    SPELL_DISARM = 22
    BIG_SLIME = 23
    SPELL_REVEAL_RATS = 24
    SPELL_REVEAL_SLIMES = 25
    GNOME = 26
    BAT = 27
    GUARD = 28
    CROWN = 29
    FIDEL = 30 # Pointless
    DRAGON_EGG = 31