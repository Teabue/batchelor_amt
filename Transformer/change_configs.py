from ruamel.yaml import YAML
import os 
# This is super ugly code, but I only wanted to implement it so that it works
# ---------------------------- SET THESE ARGUMENTS --------------------------- #
who_art_thou = 'Helena'

if who_art_thou == 'dani':
    preprocessed_data_dir = '/work3/s214629/'
    preprocessed_dataset_name = 'preprocessed_data'

    dataset_dir = '/work3/s214629/'
    dataset_rel_path_to_data_dir = ['maestro-v3.0.0/maestro-v3.0.0/2004', 
                                    'maestro-v3.0.0/maestro-v3.0.0/2006', 
                                    'maestro-v3.0.0/maestro-v3.0.0/2008',
                                    'maestro-v3.0.0/maestro-v3.0.0/2009',
                                    'maestro-v3.0.0/maestro-v3.0.0/2011',
                                    'maestro-v3.0.0/maestro-v3.0.0/2013',
                                    'maestro-v3.0.0/maestro-v3.0.0/2014',
                                    'maestro-v3.0.0/maestro-v3.0.0/2015',
                                    'maestro-v3.0.0/maestro-v3.0.0/2017',
                                    'maestro-v3.0.0/maestro-v3.0.0/2018']


    run_save_dir = '/work3/s214629/'
    run_name = 'run_a100_SOS'
    
elif who_art_thou == 'Helena':
    preprocessed_data_dir = '/work3/s214645/'
    preprocessed_dataset_name = 'preprocessed_data'

    dataset_dir = '/work3/s214645/dataset/'
    dataset_rel_path_to_data_dir = ['asap']


    run_save_dir = '/work3/s214645/'
    run_name = 'run_a100_SOS'

# --- Script will now automatically change all paths in the configs folder --- #
preprocessd_data_dir = os.path.join(preprocessed_data_dir,preprocessed_dataset_name)

dataset_dirs = [os.path.join(dataset_dir,dataset_path) for dataset_path in dataset_rel_path_to_data_dir]

run_dir = os.path.join(run_save_dir, run_name)

# Change all the files 
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open('Transformer/configs/preprocess_config.yaml', 'r') as f:
    config = yaml.load(f)

config['data_dirs'] = dataset_dirs
config['output_dir'] = preprocessd_data_dir

# Write the dictionary back to the YAML file
with open('Transformer/configs/preprocess_config.yaml', 'w') as f:
    yaml.dump(config, f)


with open('Transformer/configs/train_config.yaml', 'r') as f:
    config = yaml.load(f)
config['data_dir'] = preprocessd_data_dir
config['run_save_path'] = run_dir

with open('Transformer/configs/train_config.yaml', 'w') as f:
    yaml.dump(config, f)

print('----Done!----')