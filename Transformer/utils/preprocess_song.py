import os
import librosa
import numpy as np
import pandas as pd
import mido
import yaml
import torch

from fractions import Fraction
from music21 import *
from utils.vocabularies import Vocabulary
from .song_check import fermata_check
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
    
    
    def compute_spectrogram(self, save_path = None) -> np.ndarray:
        """Computes a ndarray representation of the spectrogram of the song and saves it as npy file.

        Raises:
            ValueError: When the preprocessing methods aren't one of the available methods

        Returns:
            np.ndarray: Spectrogram of the song
        """
        x, sr = librosa.load(self.song_path, sr=self.config['sr'])
        
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
        
        if save_path is not None:
            np.save(save_path, spectrogram)
            
        return spectrogram
        
    def preprocess_inference_new_song(self, spectrogram, cur_frame: int, bpm: int = 120) -> torch.Tensor:
        frame_times = librosa.frames_to_time(range(spectrogram.shape[1]), sr=self.config['sr'], hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
        total_duration = frame_times[-1] - frame_times[cur_frame]
        beats_left = total_duration * bpm / 60 # Assumes that the BPM is constant from here on out
        
        min_size = self.config['min_beats']
        max_size = self.config['max_beats'] if beats_left > self.config['max_beats'] else beats_left
        
        if 0 < np.round(beats_left - min_size) < max_size:
            beats_in_seq = np.round(beats_left - min_size)
        elif max_size >= min_size:
            beats_in_seq = np.random.randint(min_size, max_size + 1)
        else:
            new_frame = spectrogram.shape[1]
            return torch.from_numpy(spectrogram)[:, cur_frame:], new_frame
        
        time_to_add = beats_in_seq * 60 / bpm
        new_frame = cur_frame + np.argmin(np.abs(frame_times - time_to_add))
        
        spectrogram_slice = torch.from_numpy(spectrogram)[:, cur_frame:new_frame]
        return spectrogram_slice, new_frame
        
    def compute_labels_and_segments(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")
    
    def preprocess(self) -> None:
        NotImplementedError("This method should be implemented in the subclass")

    
class Maestro(Song):
    def __init__(self, song_filename, preprocess_config, sanity_check=False) -> None:
        self.sanity_check = sanity_check
        super().__init__(song_filename, preprocess_config)


    def compute_onset_offset_times(self, time_to_beat: bool = False):
        midi_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.midi')
        
        midi_data = mido.MidiFile(midi_path)
        note_events = []  # To store note events with onset and offset times
        ongoing_notes = {}  # To track ongoing notes
        cur_tempo = None
        cur_bpm = None
        ticks_per_beat = midi_data.ticks_per_beat 
        midi_msgs = mido.merge_tracks(midi_data.tracks)
        
        time_sec = 0.0  # Current time in secs
        time_beat = Fraction(0)
        time_ticks = 0  # Current time in ticks
        for msg in midi_msgs:
            if msg.type == 'set_tempo':
                cur_tempo = msg.tempo
                cur_bpm = mido.tempo2bpm(cur_tempo)
                continue
            
            prev_time = time_sec
            time_sec += mido.tick2second(msg.time, ticks_per_beat=ticks_per_beat, tempo=cur_tempo)  # Accumulate time
            time_beat += Fraction((time_sec - prev_time) * cur_bpm / 60)
            time_ticks += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note onset
                if msg.note not in ongoing_notes:  # Check if note is not already on
                    
                    # Store onset either in beat or seconds
                    onset = time_beat if time_to_beat else time_sec 
                    ongoing_notes[msg.note] = onset, time_ticks
                    
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                # Note offset
                if msg.note in ongoing_notes:
                    onset_time, onset_time_ticks = ongoing_notes.pop(msg.note)
                    offset = time_beat if time_to_beat else time_sec
                    
                    note_events.append({'pitch': msg.note, 
                                        'onset': onset_time, 
                                        'offset': offset,
                                        'onset_ticks': onset_time_ticks})
        # Create DataFrame
        df = pd.DataFrame(note_events)
        
        # Onset_ticks is only used to avoid rounding errors
        df_sorted = df.sort_values(by=['onset_ticks', 'pitch'], ascending=[True, True])
        
        df_final = df_sorted[['pitch', 'onset', 'offset']]
        
        return df_final
      
            
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
            
        
    def preprocess(self, **kwargs) -> None:
        spectrogram = self.compute_spectrogram(save_path = os.path.join(self.config['output_dir'], 'spectrograms', f'{self.song_name}.npy'))
        df_onset_offset = self.compute_onset_offset_times(**kwargs)
        df_labels = self.compute_labels_and_segments(df_onset_offset, spectrogram)
        return df_labels

class MuseScore(Song):
    def __init__(self, song_filename, preprocess_config, sanity_check=False) -> None:
        super().__init__(song_filename, preprocess_config)
        
        self.sanity_check = sanity_check
        
        # Allow for multiple file extensions
        for song_extension in self.config['score_file_extensions']:
            song_path_with_extension = os.path.join(os.path.dirname(self.song_path), self.song_name + '.' + song_extension)
            if os.path.isfile(song_path_with_extension):
                score_path = os.path.join(os.path.dirname(self.song_path), self.song_name + '.' + song_extension)
                self.score = converter.parse(score_path)
                break
        else:
            raise FileNotFoundError(f"No score found with extensions {self.config['score_file_extensions']} for song {self.song_name}")
        
        self.expansion_succes = False
        for parts in self.score.parts:
            exp = repeat.Expander(parts)
            expandable = exp.isExpandable()
            if (expandable is not None) and (not expandable):
                print(f"The song {self.song_name} is not expandable")
                break
        
        voltas = list(self.score.flatten().getElementsByClass('RepeatBracket'))
        if voltas:
            print(f"The song {self.song_name} has voltas")
        
        if (expandable is True or expandable is None) and not voltas:
            self.score = self.score.makeRests(fillGaps = True, inPlace = False, timeRangeFromBarDuration = True) # Some scores have missing rests and that will completely mess up the expansion
            self.score = self.score.expandRepeats()
            self.contains_fermata, self.total_duration = fermata_check(self.score)
            self.expansion_succes = True
            

    def compute_onset_offset_beats(self):
        
        # Load the MusicXML file
        df = pd.DataFrame(columns=['pitch', 'onset', 'offset']) # xml_pitch, onset time and offset time in beats
        
        # ---------------------- Add tempo ---------------------- #
        for (start_beat, end_beat, bpm) in self._get_tempos():
            df = pd.concat([df, pd.DataFrame([{'pitch': -bpm, 'onset': start_beat, 'offset': start_beat}])], ignore_index=True)
        
        # ---------------------- Add downbeats ---------------------- #
        for measure in self.score.parts[0].getElementsByClass(stream.Measure):
            # Check if it's a pickup measure (anacrusis)
            if measure.paddingLeft != 0:
                continue
            
            downbeat = measure.offset
            df = pd.concat([df, pd.DataFrame([{'pitch': -1, 'onset': downbeat, 'offset': downbeat}])], ignore_index=True)
        # Add the downbeat marking the end of the score
        # df = pd.concat([df, pd.DataFrame([{'pitch': -1, 'onset': self.score.highestTime, 'offset': self.score.highestTime}])], ignore_index=True)
        
        # ---------------------- Add notes and chords ---------------------- #  
        for element in self.score.flatten().notes:
            duration = np.array([Fraction(element.quarterLength)] * len(element.pitches))
            ele_notes = np.array([(pi.pitch.midi, pi.tie) for pi in (element.notes if element.isChord else [element])])

            # ---------------------- Handle ties ---------------------- #
            if element.tie is not None and element.tie.type == 'start':      
                next_note = element.next('Note')
                next_note_offset = next_note.offset if next_note is not None else np.inf
                next_chord = element.next('Chord')
                next_chord_offset = next_chord.offset if next_chord is not None else np.inf
                
                tied_note = next_note if next_note_offset < next_chord_offset else next_chord
                midi_and_ties = np.array([(pi.pitch.midi, pi.tie) for pi in (tied_note.notes if tied_note.isChord else [tied_note])])
                
                # Check if any of the note(s) in the next element matches any pitches with the start tied element 
                # as well as if the next element contains a stop tie. This will determine when we terminate
                pitches_matched = np.array([False if (t is not None and t.type == "start") else True for (_, t) in ele_notes])
                overlapping_pitches = np.any([[m1 == m2 and (t2 is not None and t2.type == "stop") for (m1, t1) in ele_notes] for (m2, t2) in midi_and_ties], axis = 0)
                
                # Update the duration of the found tied notes
                duration[overlapping_pitches] = duration[overlapping_pitches] + Fraction(tied_note.quarterLength)
                pitches_matched[overlapping_pitches] = True
                
                # Keep going forward in the score until we encounter a stop tie with the same pitch as the start tied element
                while not np.all(pitches_matched):
                    if isinstance(tied_note, note.Note):
                        next_note = tied_note.next('Note')
                        next_note_offset = next_note.offset if next_note is not None else np.inf # We reached the end of the score
                    else:
                        next_chord = tied_note.next('Chord')
                        next_chord_offset = next_chord.offset if next_chord is not None else np.inf
                
                    tied_note = next_note if next_note_offset < next_chord_offset else next_chord   
                    try:                     
                        midi_and_ties = np.array([(pi.pitch.midi, pi.tie) for pi in (tied_note.notes if isinstance(tied_note, chord.Chord) else [tied_note])])
                    except AttributeError:
                        raise AttributeError(f'Song {self.song_name} has a something wrong woth tie notes, getting None. Measure_number: {element.measureNumber}')
                    overlapping_pitches = np.any([[m1 == m2 and (t2 is not None and t2.type == "stop") for (m1, t1) in ele_notes] for (m2, t2) in midi_and_ties], axis = 0)
                    
                    # Update the duration of the found tied notes
                    duration[overlapping_pitches] = duration[overlapping_pitches] + Fraction(tied_note.quarterLength)
                    pitches_matched[overlapping_pitches] = True
                    
                    # Look for continuing ties and update the duration
                    continuing_pitches = np.any([[m1 == m2 and (t2 is not None and t2.type == "continue") for (m1, t1) in ele_notes] for (m2, t2) in midi_and_ties], axis = 0)
                    if np.any(continuing_pitches):
                        duration[continuing_pitches] = duration[continuing_pitches] + Fraction(tied_note.quarterLength)
            
            for i, p in enumerate(element.pitches):
                # Don't add the note if it is a tie
                if ele_notes[i, 1] is not None and ele_notes[i, 1].type in ['continue', 'stop']:
                    continue
                midi_value = p.midi
                df = pd.concat([df, pd.DataFrame([{'pitch': midi_value, 'onset': Fraction(element.offset), 'offset': Fraction(element.offset) + duration[i]}])], ignore_index=True)
        
        df = df.sort_values(by=['onset', 'pitch'])
        
        return df
    
    def compute_labels_and_segments(self, df, spectrogram, verbose = False):
        
        # ---------------------------- Calculate sequences --------------------------- #
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(range(spectrogram.shape[1]), sr=self.config['sr'], hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
        
        frame_beats = np.full_like(frame_times, fill_value = -1)
        # Calculate the onset time of all notes
        for (start_beat, start_time), (end_beat, end_time), bpm, quarter_fraction in self._merge_bpms_and_time_signatures():
            mask = (df['onset'] >= start_beat) & (df['onset'] < end_beat)
            df.loc[mask, 'onset_time'] = start_time + (df.loc[mask, 'onset'] - start_beat) * 60 / bpm
            
            # Calculate the beat of the spectrogram frames
            mask = (frame_times >= start_time) & (frame_times < end_time)
            frame_beats[mask] = start_beat + (frame_times[mask] - start_time) * bpm / 60
            
        df.loc[df.index[-1], 'onset_time'] = end_time
        
        # NOTE: Review this perhaps?
        if -1 in frame_beats:
            frame_beats[frame_beats.tolist().index(-1)] = end_beat
        
        assert df['onset_time'].isna().any() == False
        
        # Find random sequence length from min_bar to max_bar in seconds according to the bpm and time signature
        indices = [0]
        sequence_beats = [0]
        
        total_duration = self.total_duration
        min_size = self.config['min_beats']
        max_size = self.config['max_beats'] if total_duration > self.config['max_beats'] else int(total_duration)

        # Initialize a list to store the chunk sizes
        cur_beat = 0
        while max_size >= min_size:
            
            if 0 < np.round(total_duration - min_size) < max_size:
                # Makes sures that we cut the entire spectrogram by making max = min
                beats_in_seq = np.round(total_duration - min_size)
            else:
                # Generate a random size between min_size and max_size
                beats_in_seq = np.random.randint(min_size, max_size + 1)
            
            # Find the closest frame in the spectrogram and return its index
            frame = np.argmin(np.abs(frame_beats - (cur_beat + beats_in_seq)))
            
            total_duration -= (frame_beats[frame] - cur_beat)
            cur_beat = frame_beats[frame]
            
            # Gradually decrease the max_size to cut up as much as possible of the spectrogram
            if total_duration < max_size:
                max_size = np.round(total_duration)
            
            # Add the frame to the indices
            indices.extend([frame])
            sequence_beats.extend([np.round(cur_beat)]) # NOTE: This will give a little round-off error as opposed to looking up the frame_beat. However, it's necessary to preserve the grid-structure
        
        if sequence_beats[-1] != self.total_duration:
            print(f"----- {self.song_name} doesn't align properly with beats and slicing? -----")
        
        if verbose:
                
            # Plot the spectrogram along with the beats and cuts
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.title(f'{self.song_name}')
            plt.imshow(spectrogram, aspect='auto', origin='lower')
            for idx in indices:
                plt.axvline(idx, color='r', linewidth=1)
            if self.contains_fermata:
                fermata = np.argmin(np.abs(frame_beats - self.total_duration))
                plt.axvline(fermata, color='y', linewidth=1)
            plt.show()
        
        # ---------------------- Extract the sequences using onset ---------------------- #
        sequence_beats = list(zip(sequence_beats[:-1], sequence_beats[1:]))
        
        sequence_labels = []
        cur_bpm = -df.iloc[0]['pitch']
        for start_beat, end_beat in sequence_beats:
            sequence_label = df[(df['onset'] >= start_beat) & (df['onset'] < end_beat)]
            
            # Start any sequence with declaring which tempo is used
            if cur_bpm != -sequence_label.iloc[0]['pitch']:
                sequence_label = pd.concat([pd.DataFrame([{'pitch': -cur_bpm, 'onset': Fraction(start_beat), 'offset': Fraction(start_beat)}]), sequence_label], ignore_index=True)
            
            # IMPORTANT: Shift the sequence times
            sequence_label.onset = sequence_label.onset - Fraction(start_beat)
            sequence_label.offset = sequence_label.offset - Fraction(start_beat)
            
            # Find whether any tempo changes occur in this sequence
            tempos = sequence_label[sequence_label['pitch'] < -1]
            if not tempos.empty:
                cur_bpm = -tempos.iloc[-1]['pitch']
            
            sequence_labels.append(sequence_label)
            
        # ---------------------------- Translate to tokens --------------------------- #
        with open("Transformer/configs/vocab_config.yaml", 'r') as f:
            vocab_configs = yaml.safe_load(f)
        vocab = Vocabulary(vocab_configs)

        vocab.define_vocabulary(self.config['max_beats'])
        sequence_tokens = []
        df_tie_notes = None
        for (start_beat, end_beat), sequence_label in zip(sequence_beats, sequence_labels):
            sequence_duration = end_beat - start_beat
            token_sequence, df_tie_notes = vocab.translate_sequence_events_to_tokens(sequence_duration, sequence_label, df_tie_notes, self.song_name)
            sequence_tokens.append(token_sequence)
              
        # ---------------------------- Save the data --------------------------- #
        
        df = pd.DataFrame({'song_name': self.song_name,'sequence_start_idx': indices[:-1], 'sequence_end_idx': indices[1:], 'labels': sequence_tokens})
        
        return df

    def _get_tempos(self):
        # Extract tempo(s) from the score
        mm_marks = self.score.metronomeMarkBoundaries()
        
        # Make a list of list with the start and end indices of the bpm changes
        bpms_offsets, bpms = [], []
        for i, (bpm_start, bpm_end, bpm) in enumerate(mm_marks):
            if not bpms:
                bpms.append(bpm.getQuarterBPM())
                bpms_offsets.append(bpm_start)
            else:
                # During expansion, it can occur that the same bpm gets added again
                if bpms[-1] != bpm.getQuarterBPM():
                    bpms.append(bpm.getQuarterBPM())    
                    
                    # Manual time signature inserts should be omitted to avoid duplicates
                    if bpm_start not in bpms_offsets:
                        bpms_offsets.append(bpm_start)
        bpms_offsets.append(bpm_end)
        
        bpms = list(zip(bpms_offsets[:-1], bpms_offsets[1:], bpms))
        bpms = np.asarray(bpms)
        
        return bpms
    
    def _get_time_signatures(self):
        # Extract time signature(s) from the score
        time_signatures = self.score.getTimeSignatures()
        t_offsets, t_ratios = [], []
        for i, ts in enumerate(time_signatures):
            if not t_ratios:
                t_ratios.append(ts.numerator / ts.denominator)
                t_offsets.append(ts.offset)
            else:
                # During expansion, it can occur that the same time signature gets added again
                if t_ratios[-1] != ts.numerator / ts.denominator:
                    t_ratios.append(ts.numerator / ts.denominator)    
                    
                    # Manual time signature inserts should be omitted to avoid duplicates
                    if ts.offset not in t_offsets:
                        t_offsets.append(ts.offset)
        t_offsets.append(self.score.quarterLength)
        
        time_signature_offsets_and_ratios = list(zip(t_offsets[:-1], t_offsets[1:], t_ratios))
        time_signature_offsets_and_ratios = np.asarray(time_signature_offsets_and_ratios)
        
        return time_signature_offsets_and_ratios

    def _merge_bpms_and_time_signatures(self):
        
        bpms = self._get_tempos()
        time_signature_offsets_and_ratios = self._get_time_signatures()
        
        # Initialize an empty list to store the merged BPMs and time signatures
        merged = []

        # Initialize indices for the BPMs and time signatures
        bpm_index, ts_index = 0, 0

        start_time, end_time = 0, 0
        # While there are still bpms and time signatures to process
        while bpm_index < len(bpms) and ts_index < len(time_signature_offsets_and_ratios):

            bpm_start, bpm_end, bpm = bpms[bpm_index]
            ts_start, ts_end, ts = time_signature_offsets_and_ratios[ts_index]
            
            # If the bpm and time signature overlap
            if bpm_start < ts_end and ts_start < bpm_end:

                start = max(bpm_start, ts_start)
                end = min(bpm_end, ts_end)
                
                # Convert the start and end to time
                start_time = end_time
                end_time = start_time + (end - start) * 60 / bpm
                
                merged.append(((start, start_time), (end, end_time), bpm, ts))

            # If the end beat of the bpm is before the end beat of the time signature, move to the next BPM
            if bpm_end < ts_end:
                bpm_index += 1
            else:
                ts_index += 1

        return merged
            
    def preprocess(self, **kwargs) -> None:
        spectrogram = self.compute_spectrogram(save_path = os.path.join(self.config['output_dir'], 'spectrograms', f'{self.song_name}.npy'))
        df_onset_offset = self.compute_onset_offset_beats()
        df_labels = self.compute_labels_and_segments(df_onset_offset, spectrogram, **kwargs)
        return df_labels