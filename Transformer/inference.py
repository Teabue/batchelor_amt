import numpy as np
import os
import pandas as pd
import yaml

from test_inference import Inference
from utils.vocabularies import Vocabulary

from music21 import converter

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
    
    
    # Find the path to the sheet music of the song
    all_paths = [[os.path.exists(os.path.join(aud_path, f"{song}.{ext}")) for ext in configs['preprocess']['score_file_extensions']] for aud_path in audio_dirs]
    avail_paths = np.any(all_paths, axis = 1)
    if not np.any(avail_paths):
        raise FileNotFoundError(f"{song} could not be found")
    avail_ext = np.asarray(configs['preprocess']['score_file_extensions'])[np.any(all_paths, axis = 0)][0]
    song_path = os.path.join(audio_dirs[avail_paths][0], f"{song}.{avail_ext}")
    
    # Load the ground truth xml
    try:
        gt = converter.parse(song_path)
        gt = gt.makeRests(timeRangeFromBarDuration=True) # Some scores have missing rests and that will completely mess up the expansion
        gt = gt.expandRepeats()
    except:
        raise Exception(f"The ground truth file of {song} could not be succesfully processed")

    # Retrieve initial bpm
    init_bpm = gt.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
    saved_seq = os.path.exists(os.path.join(output_dir, "seq_predictions", f"{song}.npy"))
    
    # -------------------------------------- Make a prediction score  -------------------------------------------- #
    pred_score = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)

    # ------------------------------- Overlap ground truth with prediction score ------------------------------------ #
    if pred_score is not None:
        inference.overlap_gt_and_pred(gt, pred_score)