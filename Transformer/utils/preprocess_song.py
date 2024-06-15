import os
import librosa
import numpy as np
import pandas as pd
import mido
import yaml
import torch
from typing import Optional, Union
from utils.vocabularies import Vocabulary
from torch.nn.utils.rnn import pad_sequence

# Make mido ignore key signature
import mido.midifiles.meta as meta
# CAUTION: THIS WILL LITERALLY DELETE THINGS FROM THE MIDO LIBRARY - HAPPENED TO ME AT LEAST 
# del meta._META_SPECS[0x59]
# del meta._META_SPEC_BY_TYPE['key_signature']

class Song:
    def __init__(self, song_path: str | os.PathLike, preprocess_config: dict):
        """
        Args:
            song_filename (str | os.PathLike): Song filename WITH extension
            preprocess_config (dict): Config file of the preprocessing

        """
        self.song_path = song_path
        self.song_name, self.song_ext = os.path.splitext(os.path.basename(song_path)) # with extension
        self.config = preprocess_config
        os.makedirs(os.path.join(self.config['output_dir'], 'spectrograms'), exist_ok=True)
    
    
    def compute_spectrogram(self) -> np.ndarray:
        """Computes a ndarray representation of the spectrogram of the song and saves it as npy file.

        Raises:
            ValueError: When the preprocessing methods aren't one of the available methods

        Returns:
            np.ndarray: Spectrogram of the song
        """
        x, sr = librosa.load(self.song_path, sr=self.config['sr'])
        
        save_path = os.path.join(self.config['output_dir'], 'spectrograms', f'{self.song_name}.npy')
        
        if self.config['preprocess_method'] == 'cqt':
            cqt = librosa.cqt(y=x, sr=sr, hop_length=self.config['hop_length'], n_bins=self.config['n_bins'])
            cqt = librosa.amplitude_to_db(np.abs(cqt))

            spectrogram = cqt

        
        elif self.config['preprocess_method'] == 'logmel':
            mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=self.config['n_fft'], hop_length=self.config['hop_length'], n_mels=self.config['n_mels'])
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            
            spectrogram = log_mel_spectrogram
        else:
            raise ValueError("Invalid choice of preprocess method.")
        
        np.save(save_path, spectrogram)
        return spectrogram
        
    def preprocess_inference_new_song(self, sequence_length=128) -> torch.Tensor:
        spectrogram = self.compute_spectrogram()
        spectrogram = spectrogram.T
        spectrogram_slices = torch.from_numpy(spectrogram).split(sequence_length,0)

        spectrogram_slices = pad_sequence(list(spectrogram_slices), batch_first=True, padding_value=-1)
        
        return spectrogram_slices
        
    def compute_labels_and_segments(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")
    
    def preprocess(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")

    
class Maestro(Song):
    def __init__(self, song_filename, preprocess_config, sanity_check=False) -> None:
        self.sanity_check = sanity_check
        super().__init__(song_filename, preprocess_config)


    def compute_onset_offset_times(self):
        if os.path.exists(os.path.join(os.path.dirname(self.song_path), self.song_name + '.midi')):
            midi_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.midi')
        elif os.path.exists(os.path.join(os.path.dirname(self.song_path), self.song_name + '.mid')):
            midi_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.mid')
        else:
            raise FileNotFoundError(f"No .midi or .mid file found for {self.song_name}")
        
        # Load midi file to get tempo
        midi_data = mido.MidiFile(midi_path)
        note_events = []  # To store note events with onset and offset times
        ongoing_notes = {}  # To track ongoing notes
        cur_tempo = None
        ticks_per_beat = midi_data.ticks_per_beat 
        midi_msgs = mido.merge_tracks(midi_data.tracks)

        time_sec = 0.0  # Current time in secs
        time_ticks = 0  # Current time in ticks
        for msg in midi_msgs:
            if msg.type == 'set_tempo':
                cur_tempo = msg.tempo
                continue
                            
            if (cur_tempo == None and msg.time != 0) or cur_tempo != None:
                if (cur_tempo == None and msg.time != 0):
                    with open('MIDI_WO_TEMPO', 'a') as f:
                        f.write(self.song_name + '\n')
                    cur_tempo = 500000 # Set tempo to 120 bpm
                time_sec += mido.tick2second(msg.time, ticks_per_beat=ticks_per_beat, tempo=cur_tempo)  # Accumulate time
                time_ticks += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note onset
                if msg.note not in ongoing_notes:  # Check if note is not already on
                    ongoing_notes[msg.note] = time_sec, time_ticks
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                # Note offset
                if msg.note in ongoing_notes:
                    onset_time_sec, onset_time_ticks = ongoing_notes.pop(msg.note)
                    note_events.append({'pitch': msg.note, 
                                        'onset': onset_time_sec, 
                                        'offset': time_sec,
                                        'onset_ticks': onset_time_ticks})
                else:
                    raise Warning(f"Note off event without a note on event at time {time_sec}")
        # Create DataFrame
        df = pd.DataFrame(note_events)
        
        # Onset_ticks is only used to avoid rounding errors
        df_sorted = df.sort_values(by=['onset_ticks', 'pitch'], ascending=[True, True])
        
        df_final = df_sorted[['pitch', 'onset', 'offset']]
        
        return df_final
                
                
    def compute_labels_and_segments(self, df, spectrogram, sequence_length: Optional[Union[str,int]] = 'random') -> pd.DataFrame:
                
        frame_times = librosa.frames_to_time(range(spectrogram.shape[1]), sr=self.config['sr'], hop_length=self.config['hop_length'])

        # ---------------------------- Calculate sequences --------------------------- #
        min_size = 2
        max_size = self.config['max_sequence_length']
        
        
        total_size = spectrogram.shape[1]

        # Initialize a list to store the chunk sizes
        sizes = []

        # While the total size is greater than the maximum size...
        while total_size > max_size:
            # Generate a random size between min_size and max_size
            if sequence_length == 'random':
                size = np.random.randint(min_size, max_size)
            else: 
                size = sequence_length
                
            # Add the size to the list of sizes
            sizes.append(size)

            # Subtract the size from the total size
            total_size -= size

        # Add the remaining size to the list of sizes
        sizes.append(total_size)

        # Initialize start and end indices
        start_indices = np.cumsum([0] + sizes[:-1])
        end_indices = np.cumsum(sizes)

        # Combine start and end indices into a list of [start_index, end_index]
        indices = list(zip(start_indices, end_indices))
        
        sequence_times = []
        for start, end in indices:
            start_time = frame_times[start]
            
            end_time = frame_times[end - 1]

                
            sequence_times.append([start_time, end_time])
            
        sequence_times = np.array(sequence_times)
        # ---------------------- Extract the sequences using onset ---------------------- #
        sequence_labels = []

        for start_time, end_time in sequence_times:
            sequence_label = df[(df['onset'] >= start_time) & (df['onset'] < end_time)]
            
            # IMPORTANT: Shift the sequence times
            sequence_label.onset -= start_time
            sequence_label.offset -= start_time
            
            sequence_labels.append(sequence_label)
        
            
        # ---------------------------- Translate to tokens --------------------------- #
        with open("Transformer/configs/vocab_config.yaml", 'r') as f:
            vocab_configs = yaml.safe_load(f)
        vocab = Vocabulary(vocab_configs)
        
        vocab.define_vocabulary()
        
        sequence_tokens = []
        df_tie_notes = None
        for (start_time,end_time), sequence_label in zip(sequence_times, sequence_labels):
            sequence_duration = end_time - start_time
            token_sequence, df_tie_notes =  vocab.translate_sequence_events_to_tokens(sequence_duration, sequence_label, df_tie_notes)
            sequence_tokens.append(token_sequence)
              
        # ---------------------------- Save the data --------------------------- #
        
        df = pd.DataFrame({ 'song_name': self.song_name,'sequence_start_idx': start_indices, 'sequence_end_idx': end_indices,'labels': sequence_tokens})
        return df
            
        
    def preprocess(self) -> None:
        spectrogram = self.compute_spectrogram()
        df_onset_offset = self.compute_onset_offset_times()
        df_labels = self.compute_labels_and_segments(df_onset_offset, spectrogram, sequence_length=self.config['sequence_length'])
        return df_labels

