import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import copy

gamma = 0.95


class Model(nn.Module):
    def __init__(self, device=tr.device('cpu')) -> None:
        super().__init__()
        if tr.cpu.is_available() == False:
            device = tr.device('cuda')
        self.device = device

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=3, kernel_size=3, device=device)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, device=device)

        # Linear layers
        self.lin1 = nn.Linear(in_features=256, out_features=64, device=device)
        self.lin2 = nn.Linear(in_features=64, out_features=1, device=device)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        # Needs to be size [x,3,9,9] for digestion by model
        # x = number of states in sample
        if x.dim() == 3: # [3,9,9] => [1,3,9,9]
            x = x.unsqueeze(0)
        
        # CNN layers
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        
        # Flatten from matrix to array
        x = x.flatten(start_dim=1)
        
        # Linear layers
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        
        return x

    # Loss calculation according to the Bellman equation
    def loss(self, Q_value, rewards, Q_next_values, dones):
        Q_new = rewards + gamma * Q_next_values * (1 - dones)
        # If state is terminal, only count reward
        return self.mse_loss(Q_value, Q_new)

    def get_eval_values(self, next_states, dones, target):
        from classes.game_state import GameState
        values = tr.zeros((len(next_states), 1), dtype=tr.float32, device=self.device)

        # Get tensor and size list
        batch, ilist = GameState.get_afterstate_tensors(next_states, dones=dones, device=self.device)

        if batch.numel() == 0:
            return values

        # Get network and target values in 2 passes
        with tr.inference_mode():
            network_values = self(batch)
            target_values = target(batch)

        start = 0
        for i, size in ilist: # For each non-terminal state in next_state
            network_segment = network_values[start:start + size] # Start to start+len
            target_segment = target_values[start:start + size] # Start to start+len
            
            # Get best index
            if next_states[i].current_player == 1:
                best_idx = tr.argmax(network_segment).item()
            else: # Reverse values if O's perspective
                best_idx = tr.argmin(network_segment).item()
            
            values[i] = target_segment[best_idx] # Update values array
            start += size

        return values
                
    
    # Load weights from path into the model
    def load_params(self, path):
        self.load_state_dict(tr.load(path))

    # Save weights from model to path
    def save_params(self, path):
        tr.save(self.state_dict(), path)
    
    def change_device(self, device = None):
        if tr.cuda.is_available() == False:
            device = tr.device('cpu')
            return # If GPU (cuda) isn't available
        if tr.cpu.is_available() == False:
            device = tr.device('cuda')
            return # If CPU (cpu) isn't available
        if device is None:
            if self.device == tr.device('cpu'):
                self.device = tr.device('cuda')
            else:
                self.device = tr.device('cpu')
            return
        self.device = device

    def copy(self):
        return copy.deepcopy(self)