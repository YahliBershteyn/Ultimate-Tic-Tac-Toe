from .game_state import *
import random

class RandomAgent:
    def __init__(self, id=1):
        self.player_id = id
    
    def set_player_id(self, player_id): self.player_id = player_id
    
    # Choose a random legal action
    def get_action(self, state = None, epoch=0, train=None):
        moves = state.get_available_moves()
        return random.choice(moves) if moves else None
    # Epoch and train value is given as to avoid creating one code for Random/Human and one for DQN.