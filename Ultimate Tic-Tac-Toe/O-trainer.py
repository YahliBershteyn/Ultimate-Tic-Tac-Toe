from classes.game_environment import *
from classes.model_agent import *
from classes.replay_buffer import *
from classes.random_agent import *
from classes.tester import *
from classes.advanced_agent import *
import torch as tr
import os

cp_count = 10000 # The number of epochs between each checkpoint save of params
epochs = 10000000 # The overall number of epochs
start = 0 # The starting epoch
C = 100 # The number of epochs between each printout of the results
UPDATE = 15000 # The number of epochs between each update of Q_hat to Q
step_size = 3000 # The number of steps between each multiplication for the scheduler
gamma = .885 # Every step_size steps, the scheduler will multiply by gamma
# lr = 0.001 * (gamma**(start//step_size)) # Learning rate
lr = 0.001
batch_size = 64 # The number of moves fed into the model per move
env = GameEnvironment()
min_buffer = 10000 # The minimum number of moves in the buffer before any sampling can take place

RENDER = False # If you'd like the model to render changes. Advised not, purely aesthetical
WARM_START = False # Whether before each game, a warm start will be initiated

File_Num = 3 # Training id-number
INIT_NUM = 2 # Training OG-number
INIT_CP = 2 # Training OG-Checkpoint
# Creates checkpoint directory if doesn't exist
os.makedirs(f'Data/{File_Num}_checkpoints', exist_ok=True)

# Paths to files
path_load= f'Data/params_{File_Num}.pth'
init_path = f'Data/params_{INIT_NUM}.pth'
# init_path = f'Data/{INIT_NUM}_checkpoints/{INIT_CP}.pth'
buffer_path = f'Data/buffer_{File_Num}.pth'
results_path=f'Data/results_{File_Num}.pth'

def main():
    if INIT_NUM == -1:
        player = DQNAgent(id=2, path=path_load, train=True) # The agent we're training
    else: # If there is a valid init path
        player = DQNAgent(id=2, path=init_path, train=True)
    counter = AdvancedAgent(id=1) # The agent we're training our Agent against
    buffer = ReplayBuffer(path=buffer_path, batch_size=batch_size) # The replay buffer
    
    Q = player.model # Network used to choose actions
    Q_hat = Q.copy() # Network used to evaluate actions
    Q_hat.eval() # Set Q_hat not to train
    
    results_file = None
    results = []
    avglosses = []
    scores = []

    res = 0 # Reward tracker
    score = 0 # Score against Random
    loss_count = 0 # For avgLoss calculations
    avgLoss = 0 # The average loss over an epoch
    step = 0 # The number of moves
    u = 0 # Update counter
    optim = tr.optim.Adam(Q.parameters(), lr=lr) # Optimizer
    scheduler = tr.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma) # Scheduler
    
    for epoch in range(start, epochs):
        # scheduler = tr.optim.lr_scheduler.LambdaLR(optim, .95**(epoch//5000))
        print(f'epoch = {epoch}', end='\r')
        env.reset_game()
        if (WARM_START):
            env.warm_start()
        state = env.get_game_state()
        
        # Make X move and set up  O as starter
        action = counter.get_action(state)
        state.make_move(*action)
        
        # Game loop
        while not state.is_game_over:
            # Get X action
            action = player.get_action( state, epoch, True)
            as1 = state.copy()
            as1.make_move(*action)
            
            reward, done = as1.get_reward()
            res += reward
            
            if not done: # If game is not over
                # Get RND action
                action = counter.get_action(as1)
                as2 = as1.copy()
                as2.make_move(*action)
                reward2, done = as2.get_reward()
                reward -= reward2
                
                # Push moves into buffer
                buffer.push(as1, -reward, done, as2)
                state = as2
            else: # If game is over
                buffer.push(as1, -reward, done, as1)
                state = as1
                score += 2 # Compensate for -1
                if as1.game_winner != 2: # If draw
                    score -= 1
                
            # env.game_state = state
            if RENDER:
                env.render()
            
            if len(buffer) >= min_buffer:
                # Get Sampling and values
                states, rewards, dones, next_states = buffer.sample()
                
                # Get values 
                Q_values = Q(states)
                with tr.inference_mode(): # Faster no_grad()
                    Q_hat_values = Q.get_eval_values(next_states, dones, Q_hat)
                
                # Train model
                loss = Q.loss(Q_values, rewards, Q_hat_values, dones)
                optim.zero_grad()
                loss.backward()
                tr.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=10.0)
                optim.step()
                
                # Update loss
                if loss_count <= 1000:
                    avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                    loss_count += 1
                else:
                    avgLoss += (loss.item() - avgLoss) * 0.0001
            step += 1
        score -= 1
        
        # scheduler.step()
        
        # Checkpoint and update procedures
        if epoch % cp_count == 0:
            path_temp = f'Data/{File_Num}_checkpoints/{epoch//cp_count}.pth'
            Q.save_params(path_temp)
        
        if step // UPDATE == u+1: # Update Q_hat to Q and save Q
            u += 1
            Q.save_params(path_load)
            tr.save(buffer,buffer_path) # Save buffer
            Q_hat.load_state_dict(Q.state_dict())

        # Print results of episode
        if buffer.__len__() > min_buffer and (epoch+1)%C == 0:
            q0 = Q_values[0].item()
            print(f'epoch={epoch} loss={loss.item():.5f} Qvalues[0]={q0:.3f} avgloss={avgLoss:.5f} learning_rate={scheduler.get_last_lr()[0]} File={File_Num} score={score}')
            print(f'res={res:.5f}')
            # Save data to arrays
            avglosses.append(avgLoss)
            results.append(res)
            scores.append(score)
            # Update values
            res = 0
            score = 0

            # Save arrays to file
            tr.save({'epoch': epoch,'results':results,'avglosses':avglosses, 'scores':scores},results_path) # Save results  

if __name__ == '__main__':
    main()