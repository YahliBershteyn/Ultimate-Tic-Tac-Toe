from collections import deque
import random
import torch as tr
import numpy as np
from .game_state import *

class ReplayBuffer:
    def __init__(self, path=None, batch_size=64):
        if path:
            try: # If path exists
                self.buffer=tr.load(path).buffer
            except FileNotFoundError: # If path does not exist
                self.buffer=deque(maxlen=batch_size*3000)
        else: # If no path was given (path=None)
            self.buffer=deque(maxlen=batch_size*3000)
        self.batch_size = batch_size
    
    # Insert elements to front of deque
    def push(self, state, reward, done, next_state):
        self.buffer.append((state.get_board_state(), tr.tensor(reward, dtype=tr.float32), tr.tensor(done, dtype=tr.float32), next_state))
    
    # Take random sampling of the moves in the buffer
    def sample(self):
        temp = self.batch_size
        if temp > len(self.buffer):
            temp = len(self.buffer)
            
        # Get samples
        samples = random.sample(self.buffer, temp)
        # Divide samples appropriately
        state_tensors, reward_tensors, done_tensors, next_state_list = zip(*samples)
        states = tr.stack(state_tensors).to(tr.device('cpu'))
        
        # Reshapes rewards and dones for digestion by model
        rewards = tr.stack(reward_tensors).reshape(-1, 1).to(tr.device('cpu'))
        dones = tr.stack(done_tensors).reshape(-1, 1).to(tr.device('cpu'))
        next_states = list(next_state_list)
        return states, rewards, dones, next_states
    
    # Change batch_size and maxlen of self.buffer
    # After lowering batch_size, PERMENANTLY deletes data in back of deque
    def change_batch_size(self, batch_size=64):
        if self.batch_size == batch_size:
            return
        self.buffer = deque(self.buffer, maxlen=batch_size*3000)
    
    # Get size of deque
    def __len__(self):
        return len(self.buffer)