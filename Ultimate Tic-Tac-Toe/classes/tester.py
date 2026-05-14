from .random_agent import *
from .model_agent import *
from .game_environment import *

class Tester:
    def __init__(self, player1, player2, render=True):
        self.player1 = player1
        self.player2 = player2
        self.render = render
    
    # Main function
    def test(self, game_num):
        win_1, win_2, tie = 0, 0, 0 # Counts
        
        # N-Games loop
        for gn in range (game_num):
            env = GameEnvironment()
            current_player = 1
            if self.render:
                env.render()
            # Game loop
            while not env.game_state.is_game_over:
                player = self.player1 if current_player == 1 else self.player2
                action = player.get_action(env.game_state, train=False)
                
                if action is None:
                    break
                
                env.game_state.make_move(*action)
                if self.render:
                    env.render()
                current_player = 3 - current_player
            
            # Winner update
            winner = env.game_state.get_game_winner()
            if winner == 1:
                win_1 += 1
            elif winner == 2:
                win_2 += 1
            else:
                tie += 1
            
            gn += 1
        
        return win_1, win_2, tie