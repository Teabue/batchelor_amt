import os
import yaml
import pandas as pd
from utils.plots import Plotter

def get_run_losses(run_folder_paths):
    train_losses = []
    test_losses = []
    lr_epoch_starts = 1
    lr_epoch_dict = {} # Used in case learning rate was different between runs (NOTE: we could also just add lr_scheduling as normal people)
    prev_lr = None
    for run_folder in run_folder_paths:
        
        with open(os.path.join(run_folder, 'train_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            
        run_train_losses = pd.read_csv(os.path.join(run_folder, 'losses', 'train_losses.txt'))
        # TODO: IN FUTURE: CHANGE THIS TO TEST LOSSES
        run_test_losses = pd.read_csv(os.path.join(run_folder, 'losses', 'test_losses.txt'))  
        
        run_train_losses = run_train_losses[' Loss'].astype(float).tolist() # I shall sin for this ' Loss' key, but oh well
        run_test_losses = run_test_losses[' Loss'].astype(float).tolist()
        
        train_losses.extend(run_train_losses)
        test_losses.extend(run_test_losses)
        
        if prev_lr != config['lr']:
            lr_epoch_dict[config['lr']] = lr_epoch_starts
            prev_lr = config['lr']
            
        lr_epoch_starts += len(run_train_losses)
    
    return train_losses, test_losses, lr_epoch_dict 



if __name__ == '__main__':
    with open('Transformer/configs/result-visualizer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    plotter = Plotter(config['output_dir'])
    train_losses, test_losses, lr_epoch_dict = get_run_losses(config['run_folder_paths'])

    plotter.plot_loss_per_epoch(train_losses, test_losses, lr_epoch_dict)
    