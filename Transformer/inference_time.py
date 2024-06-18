import json
import numpy as np
import os
import pandas as pd
import yaml

from test_midi_inference import Inference
from utils.vocabularies import VocabTime

import os

inf_dir = "/work3/s214655/TIME_INFERENCE"
os.makedirs(inf_dir, exist_ok=True)

run_folder_name = "/work3/s214655/Time_model"

for fourier_folder in os.listdir(run_folder_name):
    outer_folder = os.path.join(inf_dir, fourier_folder)
    os.makedirs(outer_folder, exist_ok = True)
    
    run_fourier = os.path.join(run_folder_name, fourier_folder)
    for loss_folder in os.listdir(run_fourier):
        
        output_dir = os.path.join(outer_folder, loss_folder)
        os.makedirs(output_dir, exist_ok = True)
        
        run_fourier_loss = os.path.join(run_fourier, loss_folder)

        # --------------------------------- Collect configs -------------------------------- #
        with open(os.path.join(f"{run_fourier_loss}", "train_config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        
        with open(os.path.join(config['data_dir'], "preprocess_config.yaml"), 'r') as f:
            pre_configs = yaml.safe_load(f)
            
        with open(os.path.join(config['data_dir'], "vocab_config.yaml"), 'r') as f:
            vocab_configs = yaml.safe_load(f)

        configs = {"preprocess": pre_configs,
                    "vocab": vocab_configs,
                    "train": config}

        # --------------------------------- Define vocabulary -------------------------------- #
        vocab = VocabTime(vocab_configs)
        vocab.define_vocabulary()

        # ------------------------------- Choose model ------------------------------- #
        model_path = os.path.join(run_fourier_loss, "models", "model_best.pth")

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
        stat_file = os.path.join(dirs['out'], f"statistics.txt")
        if os.path.exists(stat_file):
            os.remove(stat_file)

        for song in songs:

            # Find the path to the audio of the song
            song_wo_ext =  os.path.join(data_dir, song)
            song_path = f"{song_wo_ext}.midi"

            dirs['audio'] = song_path
            
            # ----------------------------- Instantiate inference object ----------------------------- #
            inference = Inference(vocab=vocab, configs = configs, dirs = dirs, 
                                song_name = song, model_path = model_path)
            
            
            # Get the ground truth score
            gt_score, gt_events = inference.preprocess_ground_truth()
            print(gt_events)
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
            print(pred_events)
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