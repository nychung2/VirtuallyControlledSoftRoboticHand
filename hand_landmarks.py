# Labels all the hands in a List and Dictionary. Also includes a banned list for profane gestures.
hand_list = [
    'WRIST',
    'THUMB_CMC',
    'THUMB_MCP',
    'THUMB_IP',
    'THUMB_TIP',
    'INDEX_FINGER_MCP',
    'INDEX_FINGER_PIP',
    'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP',
    'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP',
    'RING_FINGER_PIP',
    'RING_FINGER_DIP',
    'RING_FINGER_TIP',
    'PINKY_MCP',
    'PINKY_PIP',
    'PINKY_DIP',
    'PINKY_TIP',
]

hand_dictionary = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5,
    'INDEX_FINGER_PIP': 6,
    'INDEX_FINGER_DIP': 7,
    'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9,
    'MIDDLE_FINGER_PIP': 10,
    'MIDDLE_FINGER_DIP': 11,
    'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13,
    'RING_FINGER_PIP': 14,
    'RING_FINGER_DIP': 15,
    'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20,
}

hand_combinations = {
    '[0, 0, 0, 0]': 'a',
    '[0, 0, 0, 1]': 'b',
    '[0, 0, 1, 0]': 'c',
    '[0, 0, 1, 1]': 'd',
    '[0, 1, 0, 0]': 'e',
    '[0, 1, 0, 1]': 'f',
    '[0, 1, 1, 0]': 'g',
    '[0, 1, 1, 1]': 'h',
    '[1, 0, 0, 0]': 'i',
    '[1, 0, 0, 1]': 'j',
    '[1, 0, 1, 0]': 'k',
    '[1, 0, 1, 1]': 'l',
    '[1, 1, 0, 0]': 'm',
    '[1, 1, 0, 1]': 'n',
    '[1, 1, 1, 0]': 'o',
    '[1, 1, 1, 1]': 'p'
}

banned_combinations = [
    '[1, 0, 1, 0, 0]', # Middle Finger 
    '[0, 0, 1, 0, 0]', # Middle Finger plus Thumb
    '[1, 0, 0, 1, 0]', # Ring Finger
    '[0, 0, 0, 1, 0]', # Ring Finger plus Thumb
]
