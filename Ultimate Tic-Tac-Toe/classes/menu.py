import pygame

class Button:
    def __init__(self, x, y, width, height, text, color=(180, 220, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color # Color of everything.
        self.selected = False
        self.font = pygame.font.Font(None, 28) # The font of the text. Size 28.

    # Draws singular button
    def draw(self, screen):
        draw_color = (70, 130, 200) if self.selected else self.color
        shadow_rect = self.rect.move(3, 3)
        pygame.draw.rect(screen, (150, 150, 150), shadow_rect, border_radius=10)
        pygame.draw.rect(screen, draw_color, self.rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2, border_radius=10)
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Due to large amount of bugs and problems, AI was used in the creation of this class.
# However, there was still much human involvement and coding.
def show_menu(screen, p1_b, p2_b, ws_b, s_b, game_winner=0):
    # Color with gradient.
    for y in range(screen.get_height()):
        color = (220 - y//10, 240 - y//10, 255 - y//10)
        pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))
    
    f_lg = pygame.font.Font(None, 48)
    f_md = pygame.font.Font(None, 36)
    f_result = pygame.font.Font(None, 42)

    # Title
    title = f_lg.render("Ultimate Tic-Tac-Toe Menu", True, (0, 0, 0))
    screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 20))

    # Print results at top of menu
    if game_winner == 1:
        result_text = "X Wins!"
        result_color = (0, 100, 200)
    elif game_winner == 2:
        result_text = "O Wins!"
        result_color = (200, 50, 0)
    elif game_winner == 3:
        result_text = "Draw!"
        result_color = (80, 80, 80)
    else:
        result_text = None

    if result_text:
        result_surf = f_result.render(result_text, True, result_color)
        screen.blit(result_surf, (screen.get_width() // 2 - result_surf.get_width() // 2, 65))

    # Labels (below result banner, above buttons)
    p1_l = f_md.render("Player X:", True, (0, 0, 0))
    screen.blit(p1_l, (100, 115))

    p2_l = f_md.render("Player O:", True, (0, 0, 0))
    screen.blit(p2_l, (550, 115))

    # Button creation
    for b in p1_b + p2_b:
        b.draw(screen)
    ws_b.draw(screen)
    s_b.draw(screen)

    pygame.display.flip()