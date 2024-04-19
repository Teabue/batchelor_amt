import os
import librosa
import numpy as np
import pandas as pd
import mido
import yaml
import torch

from music21 import *
from utils.vocabularies import Vocabulary
from torch.nn.utils.rnn import pad_sequence

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
        
    def preprocess_inference_new_song(self, sequence_size=128) -> torch.Tensor:
        spectrogram = self.compute_spectrogram()
        spectrogram = spectrogram.T
        spectrogram_slices = torch.from_numpy(spectrogram).split(128,0)

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
        midi_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.midi')
        
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
      
            
    def compute_labels_and_segments(self, df, spectrogram):
                
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
        df_labels = self.compute_labels_and_segments(df_onset_offset, spectrogram)
        return df_labels

class MuseScore(Song):
    def __init__(self, song_filename, preprocess_config, sanity_check=False) -> None:
        super().__init__(song_filename, preprocess_config)
        
        self.sanity_check = sanity_check
        
        # Allow for multiple file extensions
        for song_extension in self.config['gt_file_extensions']:
            song_path_with_extension = os.path.join(os.path.dirname(self.song_path), self.song_name + '.' + song_extension)
            if os.path.isfile(song_path_with_extension):
                xml_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.' + song_extension)
                break
        else:
            raise FileNotFoundError(f"No song found with extensions {self.config['gt_file_extensions']} at path {os.path.dirname(self.song_path)}")
        self.score = converter.parse(xml_path)

    def compute_onset_offset_beats(self):
        
        # Load the MusicXML file
        df = pd.DataFrame(columns=['pitch', 'onset', 'offset']) # xml_pitch, onset time and offset time in beats
        
        for element in self.score.flatten().notes: # NOTE: Should it perhaps be notesAndRests
            # Don't add the note if it is a tie
            if element.tie is not None and element.tie.type in ["continue", "stop"]:
                continue
            
            if isinstance(element, note.Note):
                midi_value = element.pitch.midi
                duration = element.duration.quarterLength
                
                # If the note is tied (bindebue)
                if element.tie is not None and element.tie.type in ['start', 'continue']:
                    tied_note = element.next('Note')
                    while tied_note.pitch != element.pitch and tied_note.tie is not None and tied_note.tie.type != "stop":
                        tied_note = tied_note.next('Note')
                    duration += tied_note.duration.quarterLength
                
                df = pd.concat([df, pd.DataFrame([{'pitch': midi_value, 'onset': element.offset, 'offset': float(element.offset + duration)}])], ignore_index=True)
            
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    midi_value = p.midi
                    duration = element.duration.quarterLength
                    
                    # If the note is tied (bindebue)
                    if element.tie is not None and element.tie.type in ['start', 'continue']:
                        tied_note = element.next('Chord')
                        while tied_note.pitches != element.pitches and tied_note.tie is not None and tied_note.tie.type != "stop":
                            tied_note = tied_note.next('Chord')
                        duration += tied_note.duration.quarterLength
                    
                    df = pd.concat([df, pd.DataFrame([{'pitch': midi_value, 'onset': element.offset, 'offset': float(element.offset + duration)}])], ignore_index=True)
            
            # elif isinstance(element, note.Rest):
            #     df = pd.concat([df, pd.DataFrame([{'pitch': -1, 'onset': element.offset, 'offset': float(element.offset + element.duration.quarterLength)}])], ignore_index=True)
            else:
                raise ValueError("Element is not a note or chord")
            
        return df
    
    def compute_labels_and_segments(self, df, spectrogram, bars = 1):
        # Extract tempo(s) from the score
        mm_marks = self.score.metronomeMarkBoundaries()
        
        # Make a list of list with the start and end indices of the bpm changes
        bpms = []
        for i, (bpm_start, bpm_end, bpm) in enumerate(mm_marks):
            if bpm_start == bpm_end:
                continue
            
            bpms.append([bpm_start, bpm_end, bpm.getQuarterBPM()])
        bpms = np.asarray(bpms)
        
        # Do the same with the time signatures
        time_signatures = self.score.getTimeSignatures()
        time_signature_offsets_and_ratios = []
        for i, ts in enumerate(time_signatures):
            start_beat = ts.offset
            
            if i < len(time_signatures) - 1:
                if start_beat == time_signatures[i + 1].offset:
                    continue
                end_beat = time_signatures[i + 1].offset
            else:
                end_beat = self.score.duration.quarterLength   
            
            time_signature_offsets_and_ratios.append([start_beat, end_beat, ts.numerator / ts.denominator])
        time_signature_offsets_and_ratios = np.asarray(time_signature_offsets_and_ratios)
    
        # Merge the bpms and time signatures to get a temporal overview
        bpms_and_time_sig = self._merge_bpms_and_time_signatures(bpms, time_signature_offsets_and_ratios)
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(range(spectrogram.shape[1]), sr=self.config['sr'], hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
        one_frame_in_seconds = np.diff(frame_times)[0]
        
        # Find sequence length of hyperparam "bars" in seconds according to the bpm and time signature
        indices = []
        sequence_beats = []
        
        # The spectrogram is slightly longer due to overlapping windows
        # We adjust our computed frames by doing a "time" streching of the theoretical time
        time_stretching = lambda time: (frame_times[-1] - frame_times[0]) / (bpms_and_time_sig[-1][1][1]) * (time) + frame_times[0]
        
        for (start_beat, start_time), (end_beat, end_time), bpm, quarter_fraction in bpms_and_time_sig:
            
            # Do the time alignment with the spectrogram
            spec_time_start = time_stretching(start_time)
            spec_time_end = time_stretching(end_time)
            
            # Find the frame indices of the start and end times
            start_frame = np.searchsorted(frame_times, spec_time_start)
            end_frame = np.searchsorted(frame_times, spec_time_end)
            
            spec_time_duration = spec_time_end - spec_time_start
            beat_duration = end_beat - start_beat
            no_of_bars = beat_duration / (quarter_fraction * 4)
            seconds_per_bar = spec_time_duration / no_of_bars
            
            # Calculate the slice length in frames
            slice_length = int((seconds_per_bar * bars) // one_frame_in_seconds)
            # Alternative way (should produce the same result): int((end_frame - start_frame) // no_of_bars)
            
            beats_per_bar = quarter_fraction * 4
            # seconds_per_bar = 60 / bpm * beats_per_bar
            # beats_per_second = beats_per_bar / seconds_per_bar
            
            # Calculate the slice length in frames
            # slice_length = int((seconds_per_bar * bars) // one_frame_in_seconds)
        
            # Make the slicing indices
            # start_frame = int(start_time // one_frame_in_seconds)
            # end_frame = int(end_time // one_frame_in_seconds)
            indices.extend(np.arange(start_frame, end_frame, slice_length).tolist())
            
            sequence_beats.extend(np.arange(start_beat, end_beat + 1, int(beats_per_bar * bars)).tolist())
        
        # ---------------------- Extract the sequences using onset ---------------------- #
        sequence_beats = list(zip(sequence_beats[:-1], sequence_beats[1:]))
        
        sequence_labels = []
        for start_beat, end_beat in sequence_beats:
            sequence_label = df[(df['onset'] >= start_beat) & (df['onset'] < end_beat)]
            
            # IMPORTANT: Shift the sequence times
            sequence_label.onset -= start_beat
            sequence_label.offset -= start_beat
            
            sequence_labels.append(sequence_label)
        
            
        # ---------------------------- Translate to tokens --------------------------- #
        with open("Transformer/configs/vocab_config.yaml", 'r') as f:
            vocab_configs = yaml.safe_load(f)
        vocab = Vocabulary(vocab_configs)
        
        vocab.define_vocabulary()
        sequence_tokens = []
        df_tie_notes = None
        for (start_beat, end_beat), sequence_label in zip(sequence_beats, sequence_labels):
            sequence_duration = end_beat - start_beat
            token_sequence, df_tie_notes = vocab.translate_sequence_events_to_tokens(sequence_duration, sequence_label, df_tie_notes)
            sequence_tokens.append(token_sequence)
              
        # ---------------------------- Save the data --------------------------- #
        
        df = pd.DataFrame({'song_name': self.song_name,'sequence_start_idx': indices[:-1], 'sequence_end_idx': indices[1:], 'labels': sequence_tokens})
        
        return df

    def _merge_bpms_and_time_signatures(self, bpms, time_signature_offsets_and_ratios):
        # Initialize an empty list to store the merged BPMs and time signatures
        merged = []

        # Initialize indices for the BPMs and time signatures
        bpm_index = ts_index = 0

        start_time, end_time = 0, 0
        # While there are still BPMs and time signatures to process
        while bpm_index < len(bpms) and ts_index < len(time_signature_offsets_and_ratios):
            # Get the current BPM and time signature
            bpm_start, bpm_end, bpm = bpms[bpm_index]
            
            ts_start, ts_end, ts = time_signature_offsets_and_ratios[ts_index]
            
            # If the BPM and time signature overlap
            if bpm_start < ts_end and ts_start < bpm_end:
                # The start beat is the maximum of the start beats
                start = max(bpm_start, ts_start)
                
                # The end beat is the minimum of the end beats
                end = min(bpm_end, ts_end)
                
                # Convert the start and end to time
                start_time = end_time
                end_time = start_time + (end - start) * 60 / bpm
                
                # Add the BPM and time signature to the list
                merged.append(((start, start_time), (end, end_time), bpm, ts))

            # If the end beat of the BPM is before the end beat of the time signature, move to the next BPM
            if bpm_end < ts_end:
                bpm_index += 1
            else:
                ts_index += 1

        return merged
            
    def preprocess(self) -> None:
        spectrogram = self.compute_spectrogram()
        df_onset_offset = self.compute_onset_offset_beats()
        df_labels = self.compute_labels_and_segments(df_onset_offset, spectrogram)
        return df_labels