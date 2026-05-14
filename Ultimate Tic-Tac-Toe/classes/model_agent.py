import torch as tr
import torch.nn.functional as F
import random
from .model import Model
from .game_state import GameState
import math


class DQNAgent:
    def __init__(self, id=1, path=None, train=False, device=tr.device('cpu')):
        self.model = Model(device=device)
        self.player_id = id
        self.train = train
        self.path = path
        if path is not None:
            try:
                self.model.load_params(path)
            except Exception: # If anything goes wrong
                pass

        if not self.train:
            self.model.eval()

    # Decide between training and playing
    def set_train(self, value):
        self.train = value
        if value:
            self.model.train()
        else:
            self.model.eval()
    
    # Get optimal action for state
    def get_action(self,state: GameState, epoch=1000000000, train=False):
        actions = state.get_available_moves()
        # E-greedy policy
        if self.train and train:
            epsilon = self.get_epsilon(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        # Get afterstate tensors for all actions in one vectorised call
        batch, _ = GameState.get_afterstate_tensors([state], device=self.model.device)
        
        # Get values
        with tr.inference_mode():
            values = self.model(batch)
        
        # Get best index
        if state.current_player == 1:
            best_idx = tr.argmax(values).item()
        else: # Reverse values if O's perspective
            best_idx = tr.argmin(values).item()
        
        # Return best action
        return actions[best_idx]    
    
    # save_params() of self.model
    def save_param(self, path):
        self.model.save_params(path)

    # load_params() of seld.model
    def load_params(self, path):
        self.path = path
        self.model.load_params(path)
        self.train = True
    
    # E-greedy calculation
    def get_epsilon(self,epoch,decay=25000,start=1,end=.01):
        epsilon = end+(start-end)*math.exp(-1*epoch/decay)
        return epsilon