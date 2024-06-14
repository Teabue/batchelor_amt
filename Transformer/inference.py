import json
import numpy as np
import os
import pandas as pd
import yaml

from test_inference import Inference
from utils.vocabularies import Vocabulary

# --------------------------------- Collect configs -------------------------------- #
with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
    pre_configs = yaml.safe_load(f)

with open("Transformer/configs/train_config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
# This is scuffed, but ignore it lolz
with open("Transformer/configs/vocab_config.yaml", 'r') as f:
    vocab_configs = yaml.safe_load(f)

configs = {"preprocess": pre_configs,
            "vocab": vocab_configs,
            "train": config}

# --------------------------------- Define vocabulary -------------------------------- #
vocab = Vocabulary(vocab_configs)
vocab.define_vocabulary(pre_configs['max_beats'])

# ------------------------------- Choose model ------------------------------- #
model_name = '29-05-24_musescore.pth' 

# --------------------------------- Collect directories -------------------------------- #
output_dir = "inference_songs"
data_dir = configs['preprocess']['output_dir']
audio_dirs = np.asarray(configs['preprocess']['data_dirs'])
os.makedirs(output_dir, exist_ok = True)

dirs = {"out": output_dir,
        "data": data_dir,
        "audio": audio_dirs}

# Retrieve all gt song names
df = pd.read_csv(os.path.join(data_dir, 'test', 'labels.csv'))
songs = pd.unique(df['song_name'])

# Delete statistics file if it exists
stat_file = os.path.join(dirs['out'], "statistics.txt")
if os.path.exists(stat_file):
    os.remove(stat_file)

for song in songs:
    
    # Find the path to the audio of the song
    all_paths = [[os.path.exists(os.path.join(aud_path, f"{song}.{ext}")) for ext in configs['preprocess']['audio_file_extension']] for aud_path in audio_dirs]
    avail_paths = np.any(all_paths, axis = 1)
    if not np.any(avail_paths):
        raise FileNotFoundError(f"{song} could not be found")
    avail_ext = np.asarray(configs['preprocess']['audio_file_extension'])[np.any(all_paths, axis = 0)][0]
    song_path = os.path.join(audio_dirs[avail_paths][0], f"{song}.{avail_ext}")

    dirs['audio'] = song_path
    
    # ----------------------------- Instantiate inference object ----------------------------- #
    inference = Inference(vocab=vocab, configs = configs, dirs = dirs, 
                          song_name = song, model_name = model_name)
    
    
    # Get the ground truth score
    gt = inference.preprocess_ground_truth(song)
    
    # Retrieve initial bpm
    init_bpm = gt.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    saved_seq = os.path.exists(os.path.join(output_dir, "seq_predictions", f"{song}.npy"))
    
    # -------------------------------------- Make a prediction score  -------------------------------------------- #
    pred_score = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)

    # ------------------------------- Overlap ground truth with prediction score ------------------------------------ #
    if pred_score is not None:
        inference.overlap_gt_and_pred(gt, pred_score)

# ------------------------------- Compute statistics ------------------------------------ #
# Load the stat txt file as a dataframe
stats = ["iou", "sens", "fnr", "fpr", "spec"]
df_stat = pd.read_csv(stat_file, delimiter='\t', names = stats, index_col = 0)

# Calculate confidence interval
means = df_stat.mean(axis = 0)
sems = df_stat.std(axis = 0) / np.sqrt(df_stat.shape[0])
cis = {col: [means.iloc[i] - 1.96 * sems.iloc[i], means.iloc[i] + 1.96 * sems.iloc[i]] for i, col in enumerate(df_stat.columns)}

# Write to json file
cis_file = os.path.join(dirs['out'], "cis.json")
with open(cis_file, 'w') as f:
    json.dump(cis, f, indent = 2)