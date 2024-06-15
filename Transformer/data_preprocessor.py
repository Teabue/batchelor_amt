import multiprocessing
import os 
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import cycle
from utils.preprocess_song import Maestro


import logging
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
    
    def _prepreprocess_song(self, worker_nr: int, songs: list[str,str]) -> None:
        # TODO: Make this flexible for other datasets
        
        for song_path, split in tqdm(songs, desc=f'Worker {worker_nr}', total=len(songs)):
            try:
                song = Maestro(song_path, self.config)
                df_song = song.preprocess()
                
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
                                        and (os.path.isfile(os.path.join(datafolder, os.path.splitext(file)[0] + '.midi')) 
                                        or os.path.isfile(os.path.join(datafolder, os.path.splitext(file)[0] + '.mid')))])
        
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
        with multiprocessing.Pool(self.config['num_workers']) as pool:
            pool.starmap(self._prepreprocess_song, args)
        
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
    import yaml
    import shutil 
    # Load the YAML file
    with open('Transformer/configs/preprocess_config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    # Copy over the configs used for preprocessing 
    os.makedirs(configs['output_dir'], exist_ok=True)
    shutil.copy('Transformer/configs/preprocess_config.yaml', os.path.join(configs['output_dir'], "preprocess_config.yaml"))
    shutil.copy('Transformer/configs/vocab_config.yaml', os.path.join(configs['output_dir'], "vocab_config.yaml"))

    preprocessor = DataPreprocessor(configs)
    
    preprocessor.preprocess()