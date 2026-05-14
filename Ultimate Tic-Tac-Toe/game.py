import pygame
import sys
from classes.game_environment import *
from classes.game_state import *
from classes.human_agent import HumanAgent
from classes.model_agent import DQNAgent
from classes.random_agent import RandomAgent
from classes.menu import Button, show_menu

# Make move function to make code simpler. Gets move appropriate to agent type.
def make_move1(agent, game_environment):
    state = game_environment.get_game_state()
    # Get action
    action = agent.get_action(state)
    
    # Get game result
    game_result = game_environment.make_move(action)
    if game_result != 0:
        return True, game_result
    return False, None

def main():
    # Initialization of game
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((800, 900))
    pygame.display.set_caption("Ultimate Tic-Tac-Toe")

    # Buttons
    button_width, button_height = 200, 60
    left_x = 80
    right_x = 520
    y_start = 160
    y_step = 80

    p1_buttons = [
        Button(left_x, y_start, button_width, button_height, "Human"),
        Button(left_x, y_start + y_step, button_width, button_height, "Random"),
        Button(left_x, y_start + 2 * y_step, button_width, button_height, "DQN")
    ]
    p1_buttons[0].selected = True

    p2_buttons = [
        Button(right_x, y_start, button_width, button_height, "Human"),
        Button(right_x, y_start + y_step, button_width, button_height, "Random"),
        Button(right_x, y_start + 2 * y_step, button_width, button_height, "DQN")
    ]
    p2_buttons[0].selected = True

    warm_start_button = Button(275, y_start + 3 * y_step + 20, 250, button_height, "Warm Start: ON")
    warm_start_button.selected = True

    start_button = Button(275, y_start + 4 * y_step + 60, 250, button_height, "Start Game")
    all_buttons = p1_buttons + p2_buttons + [warm_start_button, start_button]

    state = 'menu'
    running = True
    game_environment = None
    player1 = None
    player2 = None
    warm_start = True
    game_over = False
    game_winner = 0

    while running: # Game loop
        for event in pygame.event.get(): # Get all interactions with the app (Mouseclick, keyboard press)
            if event.type == pygame.QUIT: # If we stop the game, stop loop
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN: # If we click something
                if state == 'menu': # If we're in the menu
                    pos = pygame.mouse.get_pos()
                    # Check player 1 buttons
                    for i, btn in enumerate(p1_buttons):
                        if btn.is_clicked(pos):
                            for b in p1_buttons:
                                b.selected = False
                            btn.selected = True
                    # Check player 2 buttons
                    for i, btn in enumerate(p2_buttons):
                        if btn.is_clicked(pos):
                            for b in p2_buttons:
                                b.selected = False
                            btn.selected = True
                    # Check warm start
                    if warm_start_button.is_clicked(pos):
                        warm_start_button.selected = not warm_start_button.selected
                        warm_start_button.text = "Warm Start: ON" if warm_start_button.selected else "Warm Start: OFF"
                    # Check start
                    if start_button.is_clicked(pos):
                        # Set players based on selections
                        p1_type = [btn.text for btn in p1_buttons if btn.selected][0]
                        p2_type = [btn.text for btn in p2_buttons if btn.selected][0]
                        warm_start = warm_start_button.selected

                        # Menu for X
                        if p1_type == "Human":
                            player1 = HumanAgent(id=1)
                        elif p1_type == "Random":
                            player1 = RandomAgent(id=1)
                        elif p1_type == "DQN":
                            File_Num1 = 2
                            path_load1 = f'Data/params_{File_Num1}.pth'
                            player1 = DQNAgent(id=1, path=path_load1, train=False)
                        
                        # Menu for O
                        if p2_type == "Human":
                            player2 = HumanAgent(id=2)
                        elif p2_type == "Random":
                            player2 = RandomAgent(id=2)
                        elif p2_type == "DQN":
                            File_Num2 = 3
                            path_load2 = f'Data/params_{File_Num2}.pth'
                            player2 = DQNAgent(id=2, path=path_load2, train=False)

                        game_environment = GameEnvironment()
                        if warm_start:
                            game_environment.warm_start()
                        state = 'game'
                        game_over = False
                elif state == 'game' and not game_over: # If we're in the game
                    cur = game_environment.get_game_state().get_current_player()
                    current_agent = player1 if cur == 1 else player2 # Get appropriate player
                    
                    # If current player is human
                    if isinstance(current_agent, HumanAgent):
                        action = current_agent.get_action(event, game_environment)
                        
                        if action is not None: # If action exists
                            game_result = game_environment.make_move(action)
                            if game_result != 0: # If game is over
                                game_over = True
                                if game_result == 3: # If draw
                                    game_winner = 3
                                    print("Game ended in a draw!")
                                elif game_result == 1: # If X wins
                                    game_winner = cur
                                    print(f"Player {'X' if cur == 1 else 'O'} wins the game!")
                                elif game_result == 2: # If O wins
                                    game_winner = 3 - cur
                                    print(f"Player {'X' if 3 - cur == 1 else 'O'} wins the game!")
                                print("Returning to menu.")
                                state = 'menu'
            elif event.type == pygame.KEYDOWN: # If we quit game
                if event.key == pygame.K_q:
                    running = False

        if state == 'menu':
            show_menu(screen, p1_buttons, p2_buttons, warm_start_button, start_button, game_winner)
        # If game is in session.
        elif state == 'game':
            if not game_over:
                cur = game_environment.get_game_state().get_current_player()
                current_agent = player1 if cur == 1 else player2
                
                # If current player is not human
                if not isinstance(current_agent, HumanAgent):
                    game_over_flag, game_result = make_move1(current_agent, game_environment)
                    if game_over_flag:
                        game_over = True
                        cur_id = cur  # current player who just moved
                        if game_result == 3:
                            game_winner = 3
                            print("Game ended in a draw!")
                        elif game_result == 1:
                            game_winner = cur_id
                            print(f"Player {'X' if cur_id == 1 else 'O'} wins the game!")
                        elif game_result == 2:
                            game_winner = 3 - cur_id
                            print(f"Player {'X' if 3 - cur_id == 1 else 'O'} wins the game!")
                        print("Returning to menu.")
                        state = 'menu'
            
            game_environment.render() # Render new board or menu
            
            clock.tick(60) # Maxes FPS at 60

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()