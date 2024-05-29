import multiprocessing
import os 
import pandas as pd
import logging
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import cycle
from utils.preprocess_song import Maestro, MuseScore

"""
All of this is done on the cpu :^D
"""

#TODO: Bug where header does not appear, oops (easy fikx I'm just too lazy atm)

class DataPreprocessor:
    def __init__(self, preprocess_config, logger=None) -> None:
        self.config = preprocess_config
        self.logger = logger
    
    def _prepreprocess_song(self, worker_nr: int, songs: list[str,str]) -> None:
        # TODO: Make this flexible for other datasets
        
        for song_path, split in tqdm(songs, desc=f'Worker {worker_nr}', total=len(songs)):
            if self.config["dataset"] == "Maestro":
                song = Maestro(song_path, self.config)
            elif self.config["dataset"] == "MuseScore":
                song = MuseScore(song_path, self.config)
            else:
                raise ValueError("Invalid dataset")
            
            # Due to notation mistakes, expansions couldn't be resolved and the song will be omitted
            if song.expansion_succes is False:
                continue
            
            # Preprocess the song and log it if the song failed to preprocess
            try:
                df_song = song.preprocess(verbose = False)
            except Exception as e:
                if self.logger != None:
                    self.logger.error(f"Error in {song_path}: {e}")
                continue
            
            save_path = os.path.join(self.config['output_dir'], split, f'worker_{worker_nr}.csv')
            if not os.path.exists(save_path):
                first_song = True
            else:
                first_song = False
            
            df_song.to_csv(save_path, mode='a', header=first_song,index=False)
            first_song = False
            
    def preprocess(self) -> None:
        # TODO: Split up the workload through multiprocessing
        
        # Get song paths from the data directories
        song_paths = []
        for datafolder in self.config['data_dirs']:
            song_paths.extend([os.path.join(datafolder, file) for file in os.listdir(datafolder) if os.path.splitext(file)[1] in self.config['audio_file_extension']])
        
        # Get train test splits
        train, temp = train_test_split(song_paths, test_size=1-self.config['train'], random_state=self.config['seed'])
        val, test = train_test_split(temp, test_size=self.config['test']/(self.config['test'] + self.config['val']), random_state=self.config['seed'])     
        
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
    import yaml
    import shutil 
    
    # Load the YAML file
    with open('Transformer/configs/preprocess_config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    # Copy over the configs used for preprocessing 
    os.makedirs(configs['output_dir'], exist_ok=True)
    shutil.copy('Transformer/configs/preprocess_config.yaml', os.path.join(configs['output_dir'], "preprocess_config.yaml"))
    
    # shutil.copy('Transformer/configs/vocab_config.yaml', os.path.join(configs['output_dir'], "vocab_config.yaml"))

    # -------------- Small hack to change the vocab_config.yaml beat ------------- #
    from ruamel.yaml import YAML

    # Create YAML object
    yaml = YAML()
    # Load existing config
    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        vocab_configs = yaml.load(file)

    # Modify 'beat' key
    vocab_configs['event_types']['beat'] = [1, configs['max_beats'] * 12]

    # Save to new file in output directory
    with open(os.path.join(configs['output_dir'], 'vocab_config.yaml'), 'w') as file:
        yaml.dump(vocab_configs, file)
    # ------------------------------------- w ------------------------------------ #
    
    # Initialize the song logger - this will log every song that throws an error in the preprocessing
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename = os.path.join(configs['output_dir'],'song_preprocess.log'), encoding='utf-8', level=logging.ERROR)
    
    
    # Run the preprocessor
    preprocessor = DataPreprocessor(configs, logger)
    
    preprocessor.preprocess()