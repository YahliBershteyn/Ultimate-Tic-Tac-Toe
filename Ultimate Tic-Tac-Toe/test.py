from classes.model_agent import *
from classes.random_agent import *
from classes.advanced_agent import *
from classes.tester import *
File_Num = 3
Run_Num = 2
path_load= f'Data/params_{File_Num}.pth'
#path_load = f'Data/{File_Num}_checkpoints/{Run_Num}.pth'

# This is not meant for player use and so there is no menu.
# It is meant only to evaluate models and agents for the use of research.
def main():
    game_num = 100
    player2 = DQNAgent(id=2, path=path_load,train=False)
    player1 = AdvancedAgent(id=1)
    tester1 = Tester(player1, player2, False)
    result1 = tester1.test(game_num)
    print(f'P1 vs. P2:')
    print(f'  Player1 wins: {result1[0]}, Player2 wins: {result1[1]}, Ties: {result1[2]}')
    print()

if __name__ == '__main__':
    main()