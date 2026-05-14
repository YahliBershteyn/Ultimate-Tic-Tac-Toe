import random
import pygame
from .game_state import *

class HumanAgent:
    def __init__(self, id=1):
        self.player_id =  id
    
    def set_player_id(self, player_id): self.player_id = player_id

    # Used AI for debugging and fixing
    # Gets action based on position of mouse-click
    def get_action(self, event_or_pos, game_environment, train=None):
        if hasattr(event_or_pos, 'type'):
            # If event_or_pos is a mouse-click and the value of event_or_pos.button is 1 (And .button exists)
            if event_or_pos.type == pygame.MOUSEBUTTONDOWN and getattr(event_or_pos, 'button', None) == 1:
                mx, my = getattr(event_or_pos, 'pos', pygame.mouse.get_pos()) # Gets coordinates
            else:
                return None
        # If event_or_pos is a list and its length is 2 | Is coordinates
        elif isinstance(event_or_pos, (list, tuple)) and len(event_or_pos) == 2:
            mx, my = event_or_pos
        else:
            return None

        mi, ci = game_environment.pixel_to_board_position(mx, my) # Transforms coordinates to indexes
        # If indexes exist and the move is valid, return indexes
        if mi is not None and ci is not None and game_environment.get_game_state().is_valid_move(mi, ci):
            return (mi, ci)
        return None
    

    
    def get_player_id(self): return self.player_id