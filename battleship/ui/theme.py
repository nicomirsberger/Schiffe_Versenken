"""Colours and sizes of the user interface."""

BACKGROUND = (12, 27, 46)
PANEL = (20, 42, 68)
WATER = (32, 78, 120)
WATER_DARK = (24, 60, 96)
GRID_LINE = (60, 110, 155)
SHIP = (150, 160, 172)
SHIP_BORDER = (205, 212, 220)
MISS = (210, 225, 240)
HIT = (240, 120, 40)
SUNK = (200, 40, 50)
SUNK_TEXT = (255, 110, 110)  # readable on dark panels
VALID = (90, 200, 120)
INVALID = (220, 70, 70)
TEXT = (235, 240, 245)
TEXT_DIM = (150, 170, 190)
ACCENT = (255, 196, 64)
BUTTON = (40, 90, 140)
BUTTON_HOVER = (60, 120, 180)
HEAT_LOW = (40, 70, 160)
HEAT_HIGH = (255, 60, 30)

SKY_TOP = (16, 34, 58)
SEA = (22, 58, 92)
SEA_LINE = (44, 92, 138)
HULL = (112, 122, 134)
HULL_DARK = (78, 86, 96)
SUPERSTRUCTURE = (138, 148, 160)
FIRE = (255, 140, 30)
FIRE_CORE = (255, 225, 120)

WINDOW_SIZE = (1180, 840)
CELL = 36
BOARD_TOP = 290
LEFT_BOARD = (90, BOARD_TOP)
RIGHT_BOARD = (730, BOARD_TOP)
SCENE_TOP = 92  # battleship illustrations above the boards
SCENE_HEIGHT = 128
LOG_TOP = 688  # shot logs below the boards
LOG_ROWS = 3

AI_DELAY_MS = 1000  # pause before the AI's first shot of a turn
AI_REPEAT_DELAY_MS = 800  # pause between AI shots after a hit
FLASH_MS = 500
