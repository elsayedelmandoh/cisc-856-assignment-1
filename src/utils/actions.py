"""
Actions for the GridWorld environment
"""

from enum import IntEnum


class Action(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3


action_to_str = {
    Action.up : "up",
    Action.right : "right",
    Action.down : "down",
    Action.left : "left",
}

action_to_offset = {
    Action.up : (-1, 0),
    Action.right : (0, 1),
    Action.down : (1, 0),
    Action.left : (0, -1),
}