import pygame
import os
from .game_state import GameState
from .random_agent import RandomAgent

class GameEnvironment:
    
    def __init__(self, screen_width=800, screen_height=900):
        self.screen_width, self.screen_height, self.board_size = screen_width, screen_height, 720
        self.board_offset_x, self.board_offset_y = (screen_width - 720) // 2, 50 # Offset on x and y axis
        self.mini_board_size, self.cell_size = 240, 80 # Size of individual miniboards and cells
        pygame.init()

        self.screen = pygame.display.set_mode((screen_width, screen_height)) # Screen initialization
        pygame.display.set_caption("Ultimate Tic-Tac-Toe") # Title
        
        self.colors = {
            'bg': (240, 240, 240), 'grid': (100, 100, 100), 'mini_grid': (150, 150, 150),
            'active': (200, 255, 200), 'won': (255, 200, 200), 'text': (50, 50, 50)
        } # Dictionary for colors values
        
        pygame.font.init()
        self.fonts = {'lg': pygame.font.Font(None, 48), 'md': pygame.font.Font(None, 36)} # Font dictionary (Sizes)
        self.game_state = GameState() # State of the environment. Includes board, current-player, etc.
        self._load_images()
    
    # Used AI for debugging and fixing
    def _load_images(self):
        os.makedirs('images', exist_ok=True)
        
        try:
            # imgs = fixed images of X and O
            imgs = {n: self._fix_transparency(pygame.image.load(f'images/{n}.png').convert_alpha()) 
                   for n in ['x_mark', 'o_mark']}
            
            cs, bs = int(self.cell_size * 0.7), int(self.mini_board_size * 0.8)
            # Create images for cells and mini-boards
            self.image_x = pygame.transform.scale(imgs['x_mark'], (cs, cs))
            self.image_o = pygame.transform.scale(imgs['o_mark'], (cs, cs))
            self.image_win_x = self._make_semi_transparent(pygame.transform.scale(imgs['x_mark'], (bs, bs)), 180)
            self.image_win_o = self._make_semi_transparent(pygame.transform.scale(imgs['o_mark'], (bs, bs)), 180)
            
        except pygame.error: # In case of any sort of error
            self._create_images()
    
    # Used AI for creating
    def _fix_transparency(self, surf):
        w, h, fixed = surf.get_size()[0], surf.get_size()[1], pygame.Surface(surf.get_size(), pygame.SRCALPHA).convert_alpha()
        trans_cols = [(255, 255, 255), (240, 240, 240), (245, 245, 245), (248, 248, 248), (250, 250, 250), 
                     (192, 192, 192), (204, 204, 204), (220, 220, 220), (230, 230, 230), (200, 200, 200)]
        
        for x in range(w):
            for y in range(h):
                r, g, b, a = surf.get_at((x, y))
                is_trans = a < 128 or any(abs(r-tr)<=10 and abs(g-tg)<=10 and abs(b-tb)<=10 for tr,tg,tb in trans_cols)
                fixed.set_at((x, y), (0, 0, 0, 0) if is_trans else (r, g, b, a))
        
        return fixed
    
    # Used AI for creating
    def _make_semi_transparent(self, surf, alpha):
        semi = surf.copy()
        [semi.set_at((x, y), (*semi.get_at((x, y))[:3], min(alpha, semi.get_at((x, y))[3]))) 
         for x in range(surf.get_width()) for y in range(surf.get_height()) if semi.get_at((x, y))[3] > 0]
        return semi
    
    # Used AI for debugging and fixing
    def _create_images(self):
        cs, bs = int(self.cell_size * 0.7), int(self.mini_board_size * 0.8)
        
        self.image_x = pygame.Surface((cs, cs), pygame.SRCALPHA)
        m, w = cs//6, 8
        [pygame.draw.line(self.image_x, (220, 50, 50), p1, p2, w) for p1, p2 in 
         [((m, m), (cs-m, cs-m)), ((cs-m, m), (m, cs-m))]]
        
        self.image_o = pygame.Surface((cs, cs), pygame.SRCALPHA)
        pygame.draw.circle(self.image_o, (50, 50, 220), (cs//2, cs//2), cs//2-m, w)
        
        self.image_win_x = pygame.Surface((bs, bs), pygame.SRCALPHA)
        m, w = bs//10, 15
        [pygame.draw.line(self.image_win_x, (255, 0, 0, 180), p1, p2, w) for p1, p2 in 
         [((m, m), (bs-m, bs-m)), ((bs-m, m), (m, bs-m))]]
        
        self.image_win_o = pygame.Surface((bs, bs), pygame.SRCALPHA)
        pygame.draw.circle(self.image_win_o, (0, 0, 255, 180), (bs//2, bs//2), bs//2-m, w)
    
    # Translates board position to pixel coordinates
    def get_coordinates(self, mb, c=None):
        mr, mc = divmod(mb, 3)
        bx, by = self.board_offset_x + mc * self.mini_board_size, self.board_offset_y + mr * self.mini_board_size
        return (bx, by) if c is None else (bx + (c % 3) * self.cell_size, by + (c // 3) * self.cell_size)
    
    # Translates pixel-positions to miniboard and cell indexes
    def pixel_to_board_position(self, px, py):
        if not (self.board_offset_x <= px < self.board_offset_x + self.board_size and
                self.board_offset_y <= py < self.board_offset_y + self.board_size):
            return None, None
        
        rx, ry = px - self.board_offset_x, py - self.board_offset_y
        mi = (ry // self.mini_board_size) * 3 + (rx // self.mini_board_size)
        cx, cy = rx % self.mini_board_size, ry % self.mini_board_size
        ci = (cy // self.cell_size) * 3 + (cx // self.cell_size)
        return mi, ci
    
    def make_move(self, action): return self.game_state.make_move(*action)
    def get_game_state(self): return self.game_state
    def reset_game(self): self.game_state.reset()
    
    # Render board in pygame - Optional for training and testing
    def render(self):
        self.screen.fill(self.colors['bg'])
        
        text = self.fonts['lg'].render(f"Turn: {self.game_state.get_current_player_symbol()}", True, self.colors['text'])
        self.screen.blit(text, text.get_rect(center=(self.screen_width//2, 25)))
        
        [getattr(self, f'_draw_{elem}')() for elem in ['backgrounds', 'grids', 'pieces', 'win_markers']]

        pygame.event.pump()

        pygame.display.flip()
    
    # Draws background of board
    def _draw_backgrounds(self):
        ab, winners = self.game_state.get_active_mini_board(), self.game_state.get_mini_board_winners()
        
        for i in range(9):
            x, y = self.get_coordinates(i)
            color = self.colors['won'] if winners[i] != 0 else (self.colors['active'] if ab == i or ab == -1 else self.colors['bg'])
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.mini_board_size, self.mini_board_size))
    
    # Draws empty board
    def _draw_grids(self):
        # Draw the main game grid
        [pygame.draw.line(self.screen, self.colors['grid'], 
                         (self.board_offset_x + i * self.mini_board_size, self.board_offset_y), 
                         (self.board_offset_x + i * self.mini_board_size, self.board_offset_y + self.board_size), 4)
         for i in range(4)]
        [pygame.draw.line(self.screen, self.colors['grid'], 
                         (self.board_offset_x, self.board_offset_y + i * self.mini_board_size), 
                         (self.board_offset_x + self.board_size, self.board_offset_y + i * self.mini_board_size), 4)
         for i in range(4)]
        
        for mi in range(9): # For each miniboard draw its internal grid
            mx, my = self.get_coordinates(mi)
            [pygame.draw.line(self.screen, self.colors['mini_grid'], 
                             (mx + i * self.cell_size, my), (mx + i * self.cell_size, my + self.mini_board_size), 2)
             for i in range(4)]
            [pygame.draw.line(self.screen, self.colors['mini_grid'], 
                             (mx, my + i * self.cell_size), (mx + self.mini_board_size, my + i * self.cell_size), 2)
             for i in range(4)]
    
    # Draws individual X's and O's
    def _draw_pieces(self):
        board = self.game_state.board_state
        # For each miniboard
        for mi in range(9):
            # For each cell
            for ci in range(9):
                # Mark with correct value
                val = int(board[mi][ci])
                if val != 0:
                    x, y = self.get_coordinates(mi, ci)
                    # Set correct image
                    img = self.image_x if val == 1 else self.image_o
                    # Draw correct image
                    self.screen.blit(img, img.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2)))
    
    # Marks out won boards
    def _draw_win_markers(self):
        winners = self.game_state.get_mini_board_winners()
        for i in range(9): # For each mini-board, mark if won or not
            if winners[i] in [1, 2]:
                x, y = self.get_coordinates(i)
                img = self.image_win_x if winners[i] == 1 else self.image_win_o
                self.screen.blit(img, img.get_rect(center=(x + self.mini_board_size//2, y + self.mini_board_size//2)))
    
    # Initiates a warm start on the environment's board.
    def warm_start(self):
        agent = RandomAgent(id=1)
        # Completes 4 random legal moves on the board
        # Meant to ensure that no given 
        for i in range(4):
            action = agent.get_action(self.get_game_state())
            self.make_move(action)