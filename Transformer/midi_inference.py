import argparse
import json
import numpy as np
import os
import pandas as pd
import yaml

from test_midi_inference import Inference
from utils.vocabularies import VocabTime

# Create the parser
parser = argparse.ArgumentParser(description='Process Fourier and loss arguments.')

# Add mutually exclusive groups
fourier_group = parser.add_mutually_exclusive_group(required=True)
loss_group = parser.add_mutually_exclusive_group(required=True)

# Add arguments to the Fourier group
fourier_group.add_argument('--stft', action='store_true', help='Use Short-Time Fourier Transform')
fourier_group.add_argument('--cqt', action='store_true', help='Use Constant-Q Transform')
fourier_group.add_argument('--logmel', action='store_true', help='Use Log-Mel Spectrogram')

# Add arguments to the loss group
loss_group.add_argument('--cl', action='store_true', help='Use Custom Loss')
loss_group.add_argument('--ce', action='store_true', help='Use Cross-Entropy Loss')

# Parse the arguments
args = parser.parse_args()

# Determine Fourier argument
fourier_arg = 'stft' if args.stft else 'cqt' if args.cqt else 'logmel'

# Determine loss argument
loss_arg = 'cl' if args.cl else 'ce'

inf_dir = f"inference_midi_songs_{fourier_arg}_{loss_arg}"
os.makedirs(inf_dir, exist_ok=True)

run_folders = os.listdir("ablation/runs")

for run_folder in run_folders:
    
    output_dir = os.path.join(inf_dir, run_folder)
    os.makedirs(output_dir, exist_ok = True)
    
    # --------------------------------- Collect configs -------------------------------- #
    with open(os.path.join(f"{run_folder}, train_config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    with open(os.path.join(config['data_dir'], "preprocess_config.yaml"), 'r') as f:
        pre_configs = yaml.safe_load(f)
        
    with open(os.path.join(config['data_dir'], "vocab_config.yaml"), 'r') as f:
        vocab_configs = yaml.safe_load(f)

    configs = {"preprocess": pre_configs,
                "vocab": vocab_configs,
                "train": config}

    fourier = pre_configs["preprocess_method"]
    
    # --------------------------------- Define vocabulary -------------------------------- #
    vocab = VocabTime(vocab_configs)
    vocab.define_vocabulary()

    # ------------------------------- Choose model ------------------------------- #
    model_path = os.path.join(run_folder, "models", "model_best.pth")

    # --------------------------------- Collect directories -------------------------------- #
    data_dir = configs['preprocess']['output_dir']
    audio_dirs = np.asarray(configs['preprocess']['data_dirs'])

    dirs = {"out": output_dir,
            "data": data_dir,
            "audio": audio_dirs}

    # Retrieve all gt song names
    df = pd.read_csv(os.path.join(data_dir, 'test', 'labels.csv'))
    songs = pd.unique(df['song_name'])

    # Delete statistics file if it exists
    stat_file = os.path.join(dirs['out'], f"statistics-{fourier}.txt")
    if os.path.exists(stat_file):
        os.remove(stat_file)

    for song in songs:
        
        # Find the path to the audio of the song
        all_paths = [[os.path.exists(os.path.join(aud_path, f"{song}.{ext}")) for ext in configs['preprocess']['midi_file_extensions']] for aud_path in audio_dirs]
        avail_paths = np.any(all_paths, axis = 1)
        if not np.any(avail_paths):
            raise FileNotFoundError(f"{song} could not be found")
        avail_ext = np.asarray(configs['preprocess']['audio_file_extension'])[np.any(all_paths, axis = 0)][0]
        song_path = os.path.join(audio_dirs[avail_paths][0], f"{song}.{avail_ext}")

        dirs['audio'] = song_path
        
        # ----------------------------- Instantiate inference object ----------------------------- #
        inference = Inference(vocab=vocab, configs = configs, dirs = dirs, 
                            song_name = song, model_path = model_path)
        
        
        # Get the ground truth score
        gt_score, gt_events = inference.preprocess_ground_truth()
        
        # Retrieve initial bpm
        init_bpm = None
        for msg in gt_score.tracks[0]:
            if msg.type == "set_tempo":
                init_bpm = msg.tempo
                break
        
        if init_bpm is None:
            raise ValueError(f"No initial bpm found for the song: {song}")
        
        saved_seq = os.path.exists(os.path.join(output_dir, "seq_predictions", f"{song}.npy"))
        
        # -------------------------------------- Make a prediction score  -------------------------------------------- #
        pred_score, pred_events = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)

        # ------------------------------- Overlap ground truth with prediction score ------------------------------------ #
        inference.plot_pianoroll(pred_events, gt_events)

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