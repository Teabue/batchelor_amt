import argparse
import multiprocessing
import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import cycle
from utils.preprocess_song import Maestro, MuseScore
import logging
import shutil 
from ruamel.yaml import YAML

"""
All of this is done on the cpu :^D
"""
#TODO: Bug where header does not appear, oops (easy fikx I'm just too lazy atm)

# Configure logging to write to a file, setting the level to INFO
logging.basicConfig(filename='data_preprocessor.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='a')

class DataPreprocessor:
    def __init__(self, preprocess_config) -> None:
        self.config = preprocess_config
        self.dataset = self.config['dataset']
    
    def _prepreprocess_song(self, worker_nr: int, songs: list[str,str]) -> None:
        # TODO: Make this flexible for other datasets
        
        for song_path, split in tqdm(songs, desc=f'Worker {worker_nr}', total=len(songs)):
            try:
                if self.dataset == "Maestro":
                    song = Maestro(song_path, self.config)
                    df_song = song.preprocess()
                    
                elif self.dataset == "MuseScore":
                    song = MuseScore(song_path, self.config)
                    
                    # Due to notation mistakes, expansions couldn't be resolved and the song will be omitted
                    if song.expansion_succes is False:
                        continue
                    df_song = song.preprocess(verbose = False)
                    
                else:
                    raise ValueError("Invalid dataset")
                
                save_path = os.path.join(self.config['output_dir'], split, f'worker_{worker_nr}.csv')
                if not os.path.exists(save_path):
                    first_song = True
                else:
                    first_song = False
                
                df_song.to_csv(save_path, mode='a', header=first_song,index=False)
                first_song = False
            except Exception as e:
                logging.error(f"Error in {song_path}: {e}")
                continue
            
    def preprocess(self) -> None:
        # TODO: Split up the workload through multiprocessing
        
        # Get song paths from the data directories
        song_paths = []
        for datafolder in self.config['data_dirs']:
            song_paths.extend([os.path.join(datafolder, file) for file in os.listdir(datafolder) 
                                        if os.path.splitext(file)[1] in self.config['audio_file_extensions'] 
                                        and any([os.path.exists(os.path.isfile(os.path.join(datafolder, os.path.splitext(file)[0] + ext))) for ext in (self.config['midi_file_extensions'] if self.dataset == "Maestro" else self.config['score_file_extensions'])])])
        
        # Get train test splits
        train, temp = train_test_split(song_paths, test_size=1-self.config['train'], random_state=self.config['seed'])
        val, test = train_test_split(temp, test_size=self.config['test']/(self.config['test'] + self.config['val']), random_state=self.config['seed'])     
        
        # Add augmented data to training set
        if self.config['augmented_dataset']:
            # Extract base names from both validation and test files
            val_test_base_names = set(os.path.splitext(os.path.basename(file))[0] for file in val + test)
            
            # List all .midi or .mid files in the augmented_dataset directory
            augmented_files = [file for file in os.listdir(self.config['augmented_dataset']) if os.path.splitext(file)[1] in self.config['audio_file_extensions']]
            
            # Filter out files with base names that match any in the val_test_base_names set
            filtered_files = [file for file in augmented_files if not any(val_test_base in file for val_test_base in val_test_base_names)]
            
            train.extend([os.path.join(self.config['augmented_dataset'], file) for file in filtered_files])
            
        
        
        # Make output directories
        os.makedirs(os.path.join(self.config['output_dir'], 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'val'), exist_ok=True)
        
        # --------------------- Create arguments for the workers --------------------- #
        # Create a list of all songs with their corresponding split
        songs = [(song_path, split) for split, song_paths in [('train', train), ('val', val), ('test', test)]
                                    for song_path in song_paths]

        # Create a list of arguments for the workers
        args = [(worker_nr, []) for worker_nr in range(self.config['num_workers'])]

        # Distribute the songs across the workers
        for song, arg in zip(songs, cycle(args)):
            arg[1].append(song)

        # # Create a pool of workers
        # NOTE: Comment this away if debugging
        with multiprocessing.Pool(self.config['num_workers']) as pool:
            pool.starmap(self._prepreprocess_song, args)
        # self._prepreprocess_song(0, songs)
        
        # ----------------------- Concatenate the worker labels ---------------------- #
        for split in ['train', 'val', 'test']:
            csv_files = [os.path.join(self.config['output_dir'], split, file) for file in os.listdir(os.path.join(self.config['output_dir'], split)) if os.path.splitext(file)[1] == '.csv']
        
            # Read the CSV files
            dfs = [pd.read_csv(file) for file in csv_files]    

            # Delete the old CSV files
            for file in csv_files:
                os.remove(file)
                
            # Concatenate the dataframes
            df = pd.concat(dfs, ignore_index=True)

            # Write the result to a new CSV file
            df.to_csv(os.path.join(self.config['output_dir'], split,'labels.csv'))

if __name__ == '__main__':
    """ Run the file from the repo root folder"""
    
    parser = argparse.ArgumentParser(description="Update preprocess method in config.")
    parser.add_argument('--fourier', type=str, choices=['stft', 'cqt', 'logmel'], help="Specify the fourier preprocessing method") 
    parser.add_argument('--model', type=str, choices=['BeatTrack', 'TimeShift'], help="Specify the model to use: 'beat' or 'time'")
    
    
    args = parser.parse_args()
       
    # Write the updated configs back to the YAML file
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    
    # Load the YAML file
    with open('Transformer/configs/preprocess_config.yaml', 'r') as f:
        configs = y.load(f)
    
    # Load training config
    with open('Transformer/configs/train_config.yaml', 'r') as f:
        train_configs = y.load(f)
    
    # Update preprocess_method based on the argument provided
    configs['preprocess_method'] = args.fourier
    if args.fourier == "stft":
        configs['n_mels'] = 1025 # NOTE: HARDCODED
        train_configs['n_mel_bins'] = 1025 # NOTE: HARDCODED
    elif args.fourier == "cqt":
        configs['n_mels'] = 100
        train_configs['n_mel_bins'] = 100
    elif args.fourier == "logmel":
        configs['n_mels'] = 100
        train_configs['n_mel_bins'] = 100
    else:
        raise ValueError("Did not recognize fourier argument")
    
    configs['output_specific_dir'] = args.fourier
    train_configs['run_specific_path'] = args.fourier
    
    configs['output_dir'] = os.path.join(configs['output_base_dir'], configs['output_specific_dir'])
    train_configs['run_save_path'] = os.path.join(train_configs['run_base_path'], train_configs['run_specific_path'])
    train_configs['data_dir'] = os.path.join(os.path.dirname(train_configs['data_dir']), train_configs['run_specific_path'])
    
    with open('Transformer/configs/train_config.yaml', 'w') as f:
        y.dump(train_configs, f)
    
    configs['model'] = args.model
    
    if args.model == 'BeatTrack':
        configs['dataset'] = 'MuseScore'
    elif args.model == 'TimeShift':
        configs['dataset'] = 'Maestro'
    else:
        raise ValueError("Invalid model")
    
    with open('Transformer/configs/preprocess_config.yaml', 'w') as f:
        y.dump(configs, f)
    
    # Copy over the configs used for preprocessing 
    os.makedirs(configs['output_dir'], exist_ok=True)
    shutil.copy('Transformer/configs/preprocess_config.yaml', os.path.join(configs['output_dir'], "preprocess_config.yaml"))
    
    # shutil.copy('Transformer/configs/vocab_config.yaml', os.path.join(configs['output_dir'], "vocab_config.yaml"))

    # Create YAML object
    y = YAML()
    # Load existing config
    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        vocab_configs = y.load(file)

    # Modify 'beat' key
    vocab_configs['event_types']['beat'] = [1, configs['max_beats'] * 12]

    # Save to new file in output directory
    with open(os.path.join(configs['output_dir'], 'vocab_config.yaml'), 'w') as file:
        y.dump(vocab_configs, file)
    # ------------------------------------- w ------------------------------------ #
    
    preprocessor = DataPreprocessor(configs)
    
    preprocessor.preprocess()