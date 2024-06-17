import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import torch
import yaml

from collections import defaultdict
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick

from utils.vocabularies import VocabTime
from utils.preprocess_song import Song, Maestro
from utils.model import Transformer

class Inference:
    def __init__(self, vocab: VocabTime, configs: dict[str, str | None], 
                 dirs: dict[str, str | None], song_name: str, model_path: str = None):
        
        self.vocab = vocab
        
        self.pre_config = configs['preprocess']
        self.vocab_config = configs['vocab']
        self.train_config = configs['train']
        
        self.data_dir = dirs['data'] # For preprocessed data
        self.audio_dir = dirs['audio']
        self.output_dir = dirs['out']
        
        self.song_name = song_name
        self.model_path = model_path
    
    def plot_pianoroll(self, pred_events: set, gt_events: set, alpha: float = 0.5) -> None:
        
        cm_colors = {"GT": "blue", "PRED": "red", "PERF": "green"}
        
        def inference_stats(gt, pred):
            
            # Find greatest end value in both pred and gt
            gt_highest_time = max(end for _, end, _ in gt)
            pred_highest_time = max(end for _, end, _ in pred)
            
            # Compute the unions
            pred_duration = sum(end - start for start, end, _ in pred)
            gt_duration = sum(end - start for start, end, _ in gt)
            
            # Compute the total true negative area
            gt_total_area = gt_highest_time * len(np.arange(self.vocab.vocabulary['pitch'][0].min_max_values[0], self.vocab.vocabulary['pitch'][0].min_max_values[1] + 1))
            pred_total_area = pred_highest_time * len(np.arange(self.vocab.vocabulary['pitch'][0].min_max_values[0], self.vocab.vocabulary['pitch'][0].min_max_values[1] + 1))
            total_area = max(gt_total_area, pred_total_area)
            
            # Compute the intersection
            overlap_duration = sum(min(end, gt_end) - max(start, gt_start)
                            for start, end, midi in pred
                            for gt_start, gt_end, gt_midi in gt
                            if midi == gt_midi and start < gt_end and end > gt_start)
            
            union = pred_duration + gt_duration - overlap_duration
            
            false_positive = (pred_duration - overlap_duration) / total_area
            false_negative = (gt_duration - overlap_duration) / total_area
            true_positive = overlap_duration / total_area
            true_negative = (total_area - union) / total_area
            
            # Compute the relevant statistics
            iou = overlap_duration / union
            sensitivity = true_positive / (true_positive + false_negative)
            false_negative_rate = false_negative / (true_positive + false_negative)
            false_positive_rate = false_positive / (false_positive + true_negative)
            specificity = true_negative / (false_positive + true_negative)
            
            return iou, sensitivity, false_negative_rate, false_positive_rate, specificity
        
        iou, sens, fnr, fpr, spec = inference_stats(gt_events, pred_events)
        
        # Append the statistics to a TSR file
        stat_path = os.path.join(self.output_dir, f"statistics")
        with open(f"{stat_path}.txt", "a") as file:
            file.write(f"{self.song_name}\t{iou:.2f}\t{sens:.2f}\t{fnr:.2f}\t{fpr:.2f}\t{spec:.2f}\n")
        
        plt.ioff()
        _, ax = plt.subplots()

        # Find y-axis range as where notes are actually playing
        all_pitches = [midi for _, _, midi in pred_events + gt_events]
        if all_pitches:
            min_pitch = min(all_pitches)
            max_pitch = max(all_pitches)
        else:
            min_pitch, max_pitch = 21, 108  # Default range if no notes

        # Add the notes
        tp = pred_events.intersection(gt_events)
        fp = pred_events - tp
        fn = gt_events - tp
        
        for start, end, midi in [pred_events + gt_events]:
            color = cm_colors['PERF'] if (start, end, midi) in tp else cm_colors['GT'] if (start, end, midi) in fn else cm_colors['PRED']
            ax.broken_barh([(start, end)], (midi - 0.5, 1), facecolors=color, alpha=alpha)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Pitch')
        
        # Set the y-axis range to the relevant pitches
        ax.set_ylim(min_pitch - 1, max_pitch + 1)
        y_ticks = range(min_pitch, max_pitch + 1)
        ax.set_yticks(y_ticks)

        # Set y-axis labels to note names
        ax.set_yticklabels(y_ticks)

        # Add alternating background colors for rows
        for i in range(min_pitch, max_pitch, 2):
            ax.axhspan(i - 0.5, i + 0.5, facecolor='white', alpha=0.3, linewidth=0)
            ax.axhspan(i + 0.5, i + 1.5, facecolor='lightgrey', alpha=0.3, linewidth=0)

        # Add dotted horizontal lines for clarity
        for tick in y_ticks:
            ax.axhline(tick - 0.5, color='grey', linestyle='dotted', linewidth=0.5)
        
        # Combine statistics into one string
        stats_text = f"IoU: {iou:.2f}\nSensitivity: {sens:.2f}\nFalse Negative Rate: {fnr:.2f}\nFalse Positive Rate: {fpr:.2f}\nSpecificity: {spec:.2f}"

        # Define properties of the text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Adjust these values to move the position of the text box
        x_position = 1.05
        y_position = 0.7

        # Add text box
        ax.text(x_position, y_position, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        
        # Add legends to the note colors
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.plot([], [], color='blue', label='Ground Truth', alpha=alpha)
        ax.plot([], [], color='red', label='Prediction', alpha=alpha)
        ax.plot([], [], color='green', label='Perfect Prediction', alpha=alpha)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1),
                fancybox=True)

        # Save the image
        save_dir = os.path.join(self.output_dir, "piano_roll")
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, self.song_name), bbox_inches='tight')
        plt.close()
          
    def inference(self, init_bpm: int, saved_seq: bool = False, just_save_seq: bool = False) -> MidiFile:
        
        tgt_vocab_size = self.vocab.vocab_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not saved_seq:
            song_wo_ext = os.path.join(os.path.dirname(self.audio_dir), self.song_name)
            song_path = f"{song_wo_ext}.midi"
            song = Song(song_path, self.pre_config)
            
            model = Transformer(self.train_config['n_mel_bins'], tgt_vocab_size, 
                                self.train_config['d_model'], self.train_config['num_heads'], 
                                self.train_config['num_layers'], self.train_config['d_ff'], 
                                self.train_config['max_seq_length'], self.train_config['dropout'], device)
            
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.to(device)
            model.eval()

            # Prepare the sequences
            sequence_events = []
            
            # ----------------------------- Slice spectrogram dynamically ----------------------------- #
            spec_slices = song.preprocess_inference_new_song(random_slice = True)
            
            for seq_spec in spec_slices:
                # torch.tensor([[1]]) because SOS token is 1 
                output = model.get_sequence_predictions(seq_spec.unsqueeze(0), torch.tensor([[1]], device=device), self.pre_config['max_sequence_length'])
                output = output.squeeze()
                events = self.vocab.translate_sequence_token_to_events(output.tolist())
                sequence_events.append(events)
            
            os.makedirs(os.path.join(self.output_dir, 'seq_predictions'), exist_ok=True)        
            
            if sequence_events:
                np.save(os.path.join(self.output_dir, 'seq_predictions', self.song_name), sequence_events)
        
        else:
            sequence_events = np.load(os.path.join(self.output_dir, 'seq_predictions', f'{self.song_name}.npy'))
            
        if not just_save_seq:
            output_dir = os.path.join(self.output_dir, 'midi_predictions')
            os.makedirs(output_dir, exist_ok=True)   
            return self.create_midi_from_model_events(sequence_events, init_bpm, output_dir, output_name = self.song_name)
    
    def create_midi_from_model_events(translated_sequences, bpm_tempo, output_dir='', onset_only=False, output_name='output'):
        """Don't use onset_only, it's truly shit
        """
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        tempo = bpm2tempo(bpm_tempo)
        # track.append(MetaMessage('key_signature', key='A'))
        track.append(MetaMessage('set_tempo', tempo=tempo))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))
        
        for events in translated_sequences:
            delta_time = 0
            offset_onset = None
            et_processed = True
            # Check for ET first
            for event in events:
                if event[0] == 'ET':
                    et_processed = False
                    break 
                
            for event in events:
                if event[0] == 'PAD' or event[0] == 'SOS' or event[0] == 'ET':
                    if event[0] == 'ET':
                        et_processed = True
                    continue
                elif not et_processed:
                    continue
                elif event[0] == 'EOS':
                    offset_onset = None
                elif event[0] == 'time_shift':
                    delta_time += second2tick(event[1]/100, mid.ticks_per_beat, tempo)
                elif event[0] == 'offset_onset':
                    offset_onset = event[1]
                elif event[0] == 'pitch':
                    if offset_onset is None:
                        # These pitches are from before the ET token
                        continue
                    if offset_onset == 0:
                        # Just a glorified note-off event. I'm doing this because maestro did it first >:(
                        if not onset_only:
                            track.append(Message('note_off', channel=1, note=event[1], velocity=64, time=delta_time))
                    elif offset_onset == 1:
                        if onset_only:
                            track.append(Message('note_on', channel=1, note=event[1], velocity=100, time=delta_time))
                            track.append(Message('note_off', channel=1, note=event[1], velocity=100, time=delta_time+128))
                        else:
                            track.append(Message('note_on', channel=1, note=event[1], velocity=100, time=delta_time))
                    else:
                        raise ValueError('offset_onset should be 0 or 1')
                    
                    delta_time = 0

        track.append(MetaMessage('end_of_track'))

        mid.save(os.path.join(output_dir, output_name + '.midi'))
       
    def preprocess_ground_truth(self, init_bpm: int, new_song: bool = False):
        
        if new_song:
            # Find the path to the gt file
            song_wo_ext = os.path.join(os.path.dirname(self.audio_dir), self.song_name)
            song_path = f"{song_wo_ext}.midi"
            
            # Load the ground truth xml
            gt_song = Maestro(song_path, self.pre_config)
            df_gt = gt_song.preprocess()
        else:
            # Retrieve all gt song names
            df = pd.read_csv(os.path.join(self.data_dir, 'test', 'labels.csv'))
            df_gt = df[df['song_name'] == self.song_name]
        
        # Create a sheet based off of the ground truths
        sequence_events = []
        for _, row in df_gt.iterrows():
            labels = json.loads(row['labels'])
            events = self.vocab.translate_sequence_token_to_events(labels)
            sequence_events.extend(events)
        
        gt_score, gt_events = self.create_midi_from_model_events(sequence_events, init_bpm = init_bpm) 
        
        return gt_score, gt_events

if __name__ == '__main__':    
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
    vocab = VocabTime(vocab_configs)
    vocab.define_vocabulary()
    
    # ------------------------------- Choose model ------------------------------- #
    model_path = os.path.join("models", "model_best.pth")
    
    # ----------------------------- Choose song ----------------------------- #
    new_song_name = "steven_pred"
    
    # ----------------------------- Choose the type of inference ----------------------------- #
    new_song = False
    overlap = True
    
    # --------------------------------- Collect directories -------------------------------- #
    output_dir = "inference_songs"
    audio_dirs = np.asarray(configs['preprocess']['data_dirs'])
    
    all_paths = [[os.path.exists(os.path.join(aud_path, f"{new_song_name}.{ext}")) for ext in pre_configs['midi_file_extensions']] for aud_path in audio_dirs]
    avail_paths = np.any(all_paths, axis = 1)
    if not np.any(avail_paths):
        raise FileNotFoundError(f"{new_song_name} could not be found")
    avail_ext = np.asarray(pre_configs['audio_file_extension'])[np.any(all_paths, axis = 0)][0]
    song_path = os.path.join(audio_dirs[avail_paths][0], f"{new_song_name}.{avail_ext}")
    
    dirs = {"out": output_dir,
            "data": None,
            "audio": song_path}
    
    # ----------------------------- Instantiate inference object ----------------------------- #
    inference = Inference(vocab=vocab, configs = configs, dirs = dirs, 
                          song_name = new_song_name, model_path = model_path)
    
    init_bpm = 128
    
    if new_song:
        print(f"Performing inference on a new song: {new_song_name}")
        saved_seq = os.path.exists(os.path.join(dirs['out'], "seq_predictions", f"{new_song_name}.npy"))
        score = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)
       
    if overlap:

        gt, gt_events = inference.preprocess_ground_truth(init_bpm, new_song = new_song)
        
        # Get predicted score
        saved_seq = os.path.exists(os.path.join(dirs['out'], "seq_predictions", f"{new_song_name}.npy"))
        pred_score, pred_events = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)
        
        inference.plot_pianoroll(pred_events, gt_events)