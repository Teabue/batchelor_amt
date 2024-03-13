import librosa
import multiprocessing
import mido
import numpy as np
import os 
import pandas as pd
from Transformer.vocabularies import Vocabulary
"""
All of this is done on the cpu :^D
"""


class DataPreprocessor:
    def __init__(self, preprocess_config) -> None:
        self.config = preprocess_config
    
    
class Song:
    def __init__(self, song_filename, preprocess_config) -> tuple[str, np.ndarray]:
        self.song_name, self.song_ext = os.path.splitext(song_filename) # with extension
        self.config = preprocess_config
    
    
    def compute_spectrogram(self) -> str | os.PathLike:
        """Computes a ndarray representation of the spectrogram of the song and saves it as npy file.

        Raises:
            ValueError: _description_

        Returns:
            str | os.PathLike: _description_
        """
        x, sr = librosa.load(os.path.join(self.config['data_dir'], self.song_name + self.song_ext), sr=self.config['sr'])
        
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
        return save_path, spectrogram
        
        
    def compute_labels_and_segments(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")
    
    def preprocess(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")

    
class Maestro(Song):
    def __init__(self, song_filename, preprocess_config, sanity_check=False) -> None:
        self.sanity_check = sanity_check
        super().__init__(song_filename, preprocess_config)


    def compute_onset_offset_times(self):
        midi_path = os.path.join(self.config['data_dir'], self.song_name + '.midi')
        
        # Load midi file to get tempo
        midi = mido.MidiFile(midi_path)
        tempo = midi.tracks[0][0].tempo

        # Resample midifile to match a certain sample rate
        ticks_per_beat = int(tempo / (self.config['sr'] * 1e-6))
        midi = mido.MidiFile(midi_path, ticks_per_beat=ticks_per_beat)

        df = pd.DataFrame(columns=['pitch', 'onset', 'offset']) # midi_pitch, onset time and offset time in seconds
        cur_time = 0
        for msg in midi.tracks[1]:
            cur_time += mido.tick2second(msg.time, ticks_per_beat=midi.ticks_per_beat, tempo=tempo)
            if msg.type == 'note_on' and msg.velocity > 0:
                df = pd.concat([df, pd.DataFrame([{'pitch': msg.note, 'onset': cur_time}])], ignore_index=True)

            # For some god awful reason, the Maestro dataset don't use note_off events, but note_on events with velocity 0 >:((
            elif msg.type == 'note_on' and msg.velocity == 0:
                # fill out the note_off event
                df.loc[(df['pitch'] == msg.note) & (df['offset'].isnull()), 'offset'] = cur_time
        
        if self.sanity_check:
            if df['offset'].isnull().any():
                raise ValueError('Missing note_off event from Maestro preprocessing')
        
        return df
                
                
    def compute_labels_and_segments(self, df, spectrogram_path, spectrogram):
        
        
        frame_times = librosa.frames_to_time(range(spectrogram.shape[1]), sr=self.confif['sr'], hop_length=self.config['hop_length'])

        # ---------------------------- Calculate sequences --------------------------- #
        # TODO: CHANGE THIS TO A CONFIG LATER
        min_size = 2
        max_size = 100
        
        
        total_size = spectrogram.shape[1]

        # Initialize a list to store the chunk sizes
        sizes = []

        # While the total size is greater than the maximum size...
        while total_size > max_size:
            # Generate a random size between min_size and max_size
            size = np.random.randint(min_size, max_size)

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
            if end != len(frame_times):
                start_idx_next_sequence = end - 1
            else:
                start_idx_next_sequence = None # mark none so that we can just use the rest
            start_time, end_time = frame_times[start], frame_times[start_idx_next_sequence]

            sequence_times.append([start_time, end_time])
            
        sequence_times = np.array(sequence_times)
        # ---------------------- Extract the sequences using onset ---------------------- #
        sequence_labels = []

        for start_time, end_time in sequence_times:
            if end_time is None:
                # End of the song
                sequence_label = df[(df['onset'] >= start_time)]
            else:
                sequence_label = df[(df['onset'] >= start_time) & (df['onset'] < end_time)]
            sequence_labels.append(sequence_label)
        
            
        # ---------------------------- Translate to tokens --------------------------- #
        vocab = Vocabulary(self.config['vocab_config'])
        
        vocab.define_vocabulary()
        
        sequence_tokens = []
        df_tie_notes = None
        for (start_time,end_time), sequence_label in zip(sequence_times, sequence_labels):
            
            token_sequence, df_tie_notes =  vocab.translate_sequence_events_to_tokens(sequence_labels, df_tie_notes)
            sequence_tokens.append(token_sequence)
            
            
        # ---------------------------- Save the data --------------------------- #
        df = pd.DataFrame({ 'spectrogram_path': spectrogram_path,'sequence_start_idx': start_indices, 'sequence_end_idx': end_indices,'labels': sequence_tokens})
        df.to_csv(os.path.join(self.config['output_dir'], 'data.csv'), index=False)
            
        
       
        
    def preprocess(self) -> None:
        # TODO: Split up the workload through multiprocessing and lastly, add test/val/train tag on csv for the segments
        spectrogram_path, spectrogram = self.compute_spectrogram()
        df = self.compute_onset_offset_times()
        self.compute_labels_and_segments(df, spectrogram_path, spectrogram)




