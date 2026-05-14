import numpy as np
import torch as tr
import matplotlib.pyplot as plt
import os

directory = 'Data'
f_num = [1] # Training Sessions
results = []
results_path = []

for num in f_num: # For each file extract results
    results_file = f'results_{num}.pth'
    results_path.append(results_file)

for i, path in enumerate(results_path): # For each file
    full_path = os.path.join(directory, path)
    if os.path.exists(full_path): # if path exists
        data = tr.load(full_path) # Unload results into data array
        results.append(data) # Append data to results array
        # Print Results
        print(f'\n=== Results for {path} ===')
        print(f'Epoch: {data["epoch"]}')
        training_results = data['results']
        print(f'Training Results: {len(training_results)} checkpoints')
        print(f'  Max result: {max(training_results):.3f} at checkpoint {np.argmax(training_results)}')
        print(f'  Min result: {min(training_results):.3f}')
        print(f'  Mean result: {np.mean(training_results):.3f}')
        print(f'  Last result: {training_results[-1]:.3f}')
        print(f'Average Losses: {len(data["avglosses"])} entries')
        if data['avglosses']: # Filter out avgLosses + Print
            avg_losses_list = list(filter(lambda k: 0 < k < 100, data['avglosses']))
            print(f'  Max loss: {max(data["avglosses"]):.5f}')
            print(f'  Min loss: {min(data["avglosses"]):.5f}')
            print(f'  Mean loss: {np.mean(data["avglosses"]):.5f}')
            data['avglosses'] = avg_losses_list
            print(data['avglosses'][-1])
    else:
        print(f'File not found: {full_path}')

# Create graphs
for i in range(len(results)): # For each file (Group within results)
    fig, ax_list = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(results_path[i])
    plt.subplots_adjust(left=.1, right=.9, bottom=.1, top=.9)
    
    # Graph A - Results
    ax_list[0].plot(results[i]['results'])
    ax_list[0].set_title('Training Results')
    ax_list[0].set_ylabel('Result')
    ax_list[0].set_xlabel('Training Step')
    
    # Graph B - Average Losses
    ax_list[1].plot(results[i]['avglosses'])
    ax_list[1].set_title('Average Loss')
    ax_list[1].set_ylabel('Loss')
    ax_list[1].set_xlabel('Training Step')
    
    # Graph C - Scores
    ax_list[2].plot(results[i]['scores'])
    ax_list[2].set_title('Training Scores')
    ax_list[2].set_ylabel('Score')
    ax_list[2].set_xlabel('Training Step')
    
    # Print Graphs
    plt.tight_layout()

plt.show()        