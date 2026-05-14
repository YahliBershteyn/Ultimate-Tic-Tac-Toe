import numpy as np
import torch as tr

board_indexes = np.array([
     0,  1,  2,  9, 10, 11, 18, 19, 20,
     3,  4,  5, 12, 13, 14, 21, 22, 23,
     6,  7,  8, 15, 16, 17, 24, 25, 26,
    27, 28, 29, 36, 37, 38, 45, 46, 47,
    30, 31, 32, 39, 40, 41, 48, 49, 50,
    33, 34, 35, 42, 43, 44, 51, 52, 53,
    54, 55, 56, 63, 64, 65, 72, 73, 74,
    57, 58, 59, 66, 67, 68, 75, 76, 77,
    60, 61, 62, 69, 70, 71, 78, 79, 80,
], dtype=np.int32)


class GameState:
    def __init__(self, board=None, mini_board_winners=None, cur=None, active=None, won_mini=False):
        if board is None:    
            self.board_state = np.zeros((9, 9), dtype=int) # Board = 9x9 zeroes [mb,c]
            self.mini_board_winners = np.zeros(9, dtype=int) # Mini-board winners
            self.current_player, self.active_mini_board, self.game_winner, self.is_game_over = 1, -1, 0, False
            self.won_mini = False
            self.two_win = False
        else:
            self.board_state = board
            self.mini_board_winners = mini_board_winners
            self.current_player=cur
            self.active_mini_board=active
            self.game_winner, self.is_game_over = 0, False
            self.won_mini = won_mini
            self.two_win = False
    
    def get_mini_board_winners(self): return self.mini_board_winners.copy()
    def get_current_player(self): return self.current_player
    def get_current_player_symbol(self): return 'X' if self.current_player == 1 else 'O'
    def get_active_mini_board(self): return self.active_mini_board
    def get_game_winner(self): return self.game_winner
    
    # Checks if given move is valid - Used for game.py mostly
    def is_valid_move(self, mini_board_index, cell_index):
        return not self.is_game_over and self.mini_board_winners[mini_board_index] == 0 and \
               self.board_state[mini_board_index][cell_index] == 0 and \
               (self.active_mini_board == -1 or self.active_mini_board == mini_board_index)
    
    # Marks point [mb,c] for current player.
    # Checks for mini-board winners and overall winners
    def make_move(self, mb, c):
        if not self.is_valid_move(mb, c):
            return 0
        self.board_state[mb][c] = self.current_player
        mini_won = self._check_mini_board_winner(mb, c)
        if mini_won:
            self.won_mini = True
        self.active_mini_board = c if self.mini_board_winners[c] == 0 else -1
        self._check_game_winner()
        self.current_player = 3 - self.current_player
        if self.game_winner == 3-self.current_player:
            return 1
        if self.game_winner == self.current_player:
            return 2
        if self.game_winner == 3:
            return 3
        return 0
    
    # Used AI for debugging and fixing
    def _check_mini_board_winner(self, mini_board_index, cell_index):
        board = self.board_state[mini_board_index].reshape(3, 3)
        row = cell_index // 3
        col = cell_index - row * 3

        lines = [board[row, :], board[:, col]]
        # If cell is corner or middle
        if row == col:
            lines.append(np.diag(board))
        if row + col == 2:
            lines.append(np.diag(np.fliplr(board)))

        self.two_win = False

        # Check if won mini-board
        for line in lines:
            if line[0] == line[1] == line[2] != 0:
                self.mini_board_winners[mini_board_index] = line[0]
                return True

        # Check if won 2 in a row
        current = self.current_player
        for line in lines:
            if np.count_nonzero(line == current) == 2 and np.count_nonzero(line == 0) == 1:
                self.two_win = True

        # Checks if mini-board is completely full.
        if np.all(self.board_state[mini_board_index] != 0):
            self.mini_board_winners[mini_board_index] = 3
            return True
        
        return False
    
    # Checks if any player has won or lost, or if the game is drawn
    def _check_game_winner(self):
        winners = self.mini_board_winners.reshape(3, 3)
        lines = [winners[i] for i in range(3)] + [winners[:, i] for i in range(3)] + \
                [np.diag(winners), np.diag(np.fliplr(winners))]
        
        for line in lines: # For all possible mini-board lines
            if line[0] == line[1] == line[2] != 0 and line[0] != 3:
                # If line is a win
                self.game_winner, self.is_game_over = line[0], True
                return True
        
        if np.all(self.mini_board_winners != 0):
            # If all mini-boards are won and no player has won
            self.game_winner, self.is_game_over = 3, True
            # Return draw
            return True
        
        return False
    
    # Resets the board
    def reset(self):
        self.board_state = np.zeros((9, 9), dtype=int)
        self.mini_board_winners = np.zeros(9, dtype=int)
        self.current_player, self.active_mini_board, self.game_winner, self.is_game_over = 1, -1, 0, False
        self.won_mini = False
        self.two_win = False
    
    # Returns a list of legal moves in the form of [mb,c] index
    def get_available_moves(self):
        if self.is_game_over: return []
        
        # Get list of valid mini-boards get legal actions
        valid_boards = [i for i in range(9) if self.mini_board_winners[i] == 0 and 
                       (self.active_mini_board == -1 or self.active_mini_board == i)]
        
        # Return list of all empty cells in legal mini-boards
        return [(mb, c) for mb in valid_boards for c in range(9) if self.board_state[mb][c] == 0]
    
    # Returns the proper state of a board for the model to digest
    def get_board_state(self, device=tr.device('cpu')):
        # --- channel 0: piece positions (player 1 → +1, player 2 → -1) ---
        state_flat = self.board_state.flatten().astype(np.float32)
        state_flat[state_flat == 2] = -1.0

        # --- channel 1: legal-move mask (-1 at each legal (mb, c) cell) ---
        legal_flat = np.zeros(81, dtype=np.float32)
        moves = self.get_available_moves()
        if moves:
            mbs, cs = zip(*moves)
            legal_flat[np.array(mbs) * 9 + np.array(cs)] = -1.0

        # --- channel 2: mini-board winner overlay ---
        # Map winner values → display values: 1→1, 2→-1, 3→2, 0→0
        w = self.mini_board_winners                          # shape (9,)
        vals = np.where(w == 1, 1.0,
               np.where(w == 2, -1.0,
               np.where(w != 0,  2.0, 0.0))).astype(np.float32)
        mini_win_flat = np.repeat(vals, 9)                  # shape (81,)

        # --- rearrange all three channels from [mb*9+c] to [row*9+col] ---
        out = np.empty((3, 81), dtype=np.float32)
        out[0, board_indexes] = state_flat
        out[1, board_indexes] = legal_flat
        out[2, board_indexes] = mini_win_flat

        return tr.tensor(out.reshape(3, 9, 9), dtype=tr.float32, device=device)
    
    @staticmethod
    def get_afterstate_tensors(states, dones=None, device=tr.device('cpu')):
        tensors_list = []
        ilist = [] # [index, length]

        for i, state in enumerate(states): # For each state
            # If terminal, skip
            if dones is not None and dones[i].item():
                continue
            if state.is_game_over:
                continue

            actions = state.get_available_moves()
            if not actions:
                continue

            count = len(actions)
            # Build afterstates with no loops
            def _make_afterstate(action, _state=state, _dev=device):
                s = _state.copy()
                s.make_move(*action)
                return s.get_board_state(device=_dev)
            tensors_list += list(map(_make_afterstate, actions))

            ilist.append((i, count))

        if not tensors_list: # If tensors_list is empty
            return tr.empty(0, 3, 9, 9, dtype=tr.float32, device=device), ilist

        return tr.stack(tensors_list), ilist
    
    # Returns a copy (Not reference) of the current board
    def copy(self):
        newBoard = np.copy(self.board_state)
        newMini = np.copy(self.mini_board_winners)
        newCur = np.copy(self.current_player)
        newActive = np.copy(self.active_mini_board)
        new_state = GameState(newBoard, newMini, newCur, newActive, self.won_mini)
        new_state.game_winner = self.game_winner
        new_state.is_game_over = self.is_game_over
        new_state.two_win = self.two_win
        return new_state
    
    # Immediete reward function
    def get_reward(self):
        reward=0
        AWR = 1 # Overall victory
        FB = .05 # Free board penalty
        TW = .03 # Two in a row reward
        MW = .16 # Mini-board win reward
        done=self.is_game_over
        if self.active_mini_board == -1 and done == False and self.won_mini == False:
            reward -= FB
        if self.won_mini:
            reward += MW
            self.won_mini=False
        if self.two_win:
            reward += TW
            self.two_win = False
        if done:
            if self.game_winner == 3 - self.current_player:
                reward += AWR
            elif self.game_winner in [0,3]:
                reward -= 0.5*AWR
            else:
                reward -= AWR

        return reward,done