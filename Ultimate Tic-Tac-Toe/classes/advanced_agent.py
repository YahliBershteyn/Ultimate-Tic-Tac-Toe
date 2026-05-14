from .game_state import *
import numpy as np

class AdvancedAgent:
    def __init__(self, id=1):
        self.player_id = id
    
    def set_player_id(self, player_id): self.player_id = player_id
    
    # Exact same function as that of DQNAgent, only without epsilon-greedy.
    def get_action(self,state: GameState, epoch=0, train=False, temp=.5):
        actions = state.get_available_moves()
        rewards = []
        
        # For each legal action
        for action in actions:
            # Create afterstate
            afterstate = state.copy()
            afterstate.make_move(*action)
            
            # Evaluate afterstate
            value, done = afterstate.get_reward()
            rewards.append(value)
        
        rewards = np.array(rewards, dtype=np.float32)
        rewards = rewards - np.max(rewards) # Regulate scores
        exp = np.exp(rewards / temp) # Get e^values
        probs = exp / np.sum(exp) # Get softmax probabilities
        
        # Get action by reward softmax
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]    