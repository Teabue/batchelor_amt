"""
Structure of the vocabulary:

The vocabulary containes several eventtypes
Keep a dictionairy of what ranges of tokens are what eventtype
The use that event.translate_from_token(token) to get the event value.
"""
import pandas as pd
import numpy as np

from fractions import Fraction

# TODO: Add time shift to the vocabulary in a way that makes sense :^)
# TODO: Loading tuples from a yaml file is irritating >:(
# TODO: I'm 100% sure all of this could be optimized with at least 200% :^)
    
class EventType:
    """
    Event
    """
    def __init__(self, event_type: str, min_max_values: tuple[int, int]):
        self.event_type = event_type 
        self.min_max_values = min_max_values 
        
    def set_token_range(self, start_token):
        self.token_start = start_token
        range_width = self.min_max_values[1] - self.min_max_values[0]
        self.end_token = start_token + range_width
        return self.end_token + 1 
    
    def translate_token_to_value(self, token):
        if not (token >= self.token_start and token <= self.end_token):
            raise ValueError('Token out of range')
        
        return int(token - self.token_start + self.min_max_values[0]) # NOTE: Assume only integer values for now
    
    def translate_value_to_token(self, value, song_name: str = 'Song_name_not_given'):
        if not (value >= self.min_max_values[0] and value <= self.min_max_values[1]):
            raise ValueError(f'[ERROR] Vocabulary: Value {value} out of range for event {self.event_type} for song {song_name}')
        
        return int(value - self.min_max_values[0] + self.token_start)
        
        
class Vocabulary:
    def __init__(self, config, dtype):
        self.config = config
        self.type = dtype
    
    
    def _define_event(self, token_offset, token_type, min_max_values):
        min_token = token_offset
        event = EventType(token_type, min_max_values)
        token_offset = event.set_token_range(token_offset)
        self.vocabulary[token_type] = (event, (min_token, token_offset - 1))
        return token_offset
    
    
    def define_vocabulary(self, h_bars=None):
        self.vocabulary: dict[EventType, tuple[int,int]] = {} 
        token_offset = 0
        
        # Set special tokens
        for special_token in self.config['special_tokens']:
            
            # Time shift model shouldn't have downbeat in the vocab
            if special_token == 'downbeat':
                if self.type == "time":
                    continue
            
            token_offset = self._define_event(token_offset, special_token, (0,0))

        for event_type, min_max_values  in self.config['event_types'].items():
            # Small hack for adjusting beat tokens depending on h_bars
            if event_type == 'beat' and min_max_values == 'None':
                if self.type == "beat":
                    min_max_values = [1, h_bars * 12]
                else:
                    # Time shift shouldn't have beat in the vocab
                    continue
            
            # Beat track model shouldn't have time shift in the vocab
            elif event_type == 'time_shift':
                if self.type == "beat":
                    continue
            
            # Time shift shouldn't have tempo or downbeat in the vocab
            elif event_type == 'tempo':
                if self.type == "time":
                    continue
                
            token_offset = self._define_event(token_offset, event_type, min_max_values)

        self.vocab_size = token_offset 
    
    def translate_sequence_token_to_events(self, sequence: list[int]) -> list[tuple[str, int]]:
        """Translates a sequence of tokens to a sequence of events
        Used only in inference to create midi 
        Args:
            sequence (list[int])
        """
        # NOTE: This is probably a really slow way to do it, alternative implementation up for consideration
        # The input will even prolly be a tensor :^)
        
        translated_sequence = []
        for token in sequence:
            
            for event, (event, token_range) in self.vocabulary.items():
                if token >= token_range[0] and token <= token_range[1]:
                    event_value = event.translate_token_to_value(token)
                    translated_sequence.append((event.event_type, event_value))
                    break
        return translated_sequence
    
    def translate_sequence_events_to_tokens(self, sequences: list[pd.DataFrame], song_name: str = 'Song_name_not_given') -> list[int]:
        raise NotImplementedError("This function should be implemented in a subclass")


class VocabTime(Vocabulary):
    def __init__(self, config):
        super().__init__(config, dtype = "time")

    def translate_sequence_events_to_tokens(self, duration, df_sequence: pd.DataFrame, df_tie_notes: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
        """See Transformer/docs/vocabulary.md for more info 
        
        Args:
            sequences (list[DataFrame]): Assumes for now a dataframe with pitch(int), onset(float), offset(float)
        """        
        # ---------------- In case no notes are played in the sequence --------------- #
        if len(df_sequence) == 0:
            duration_in_ten_ms = np.floor(duration * 100)
            token_sequence = []
            token_sequence.append(self.vocabulary['SOS'][0].translate_value_to_token(0))
            token_sequence.append(self.vocabulary['time_shift'][0].translate_value_to_token(duration_in_ten_ms))
            token_sequence.append(self.vocabulary['EOS'][0].translate_value_to_token(0))
            return token_sequence, None
        
        # ------------------------------ Setup dataframe ----------------------------- #
        # Unravel the dataframe to have columns (pitch, type, time) where type is either onset or offset
        df_sequence = df_sequence.melt(id_vars='pitch', value_vars=['onset', 'offset'], var_name='type', value_name='time')
        #thank god for copilot :^^DD
        
        # Compute tie notes where the offset is greater than the duration
        df_tie_notes = df_sequence[(df_sequence['type'] == 'offset') & (df_sequence['time'] > duration)]
        if len(df_tie_notes) == 0: # If there are no tie notes, set it to None
            df_tie_notes = None
            
        # Remove these rows from the sequence we want to tokenize
        df_sequence = df_sequence[~((df_sequence['type'] == 'offset') & (df_sequence['time'] > duration))]
        
        # Sort the sequence by time
        df_sequence = df_sequence.sort_values(by='time')
        
        # TODO: MAKE THIS FLEXIBLE
        # Round to nearest 10 ms (0.01s) and convert to per 10 ms
        df_sequence['time'] = (pd.to_numeric(df_sequence['time']) * 100).round() 
        
        # If any of the rounded times exceeded the duration, subtract 1
        duration_in_ten_ms = duration * 100
        exceeded_max = df_sequence['time'] > duration_in_ten_ms
        df_sequence.loc[exceeded_max, 'time'] = df_sequence.loc[exceeded_max, 'time'] - 1
        
        last_offset_onset_value = None
        token_sequence = []
        # Add the start of sequence token
        token_sequence.append(self.vocabulary['SOS'][0].translate_value_to_token(0))

        # ----------------------------- Declare tie notes ----------------------------- #
        if df_tie_notes is not None:
            df_tie_notes = df_tie_notes.sort_values(by='pitch')
            
            for _, row in df_tie_notes.iterrows():
                token_sequence.append(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']))

            token_sequence.append(self.vocabulary['ET'][0].translate_value_to_token(0))
        
        # ------------------------ Get the rest of the labels ------------------------ #
                
        cur_time = 0
        # First group by time
        for time_value, group in df_sequence.groupby('time'):
            # Add time shift
            if time_value - cur_time != 0: # This statement should only be used for the first time shift
                token_sequence.append(self.vocabulary['time_shift'][0].translate_value_to_token(time_value - cur_time))
            # Update current time
            cur_time = time_value
            
            # OH GAWD MY EYES ARE BURNING; REFACTOR THIS AT ALL COSTS AAAAaaa
            if last_offset_onset_value is None:
                token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(1)) # start with onset
                last_offset_onset_value = 1
            
            onset_rows = group.loc[group['type'] == 'onset']
            offset_rows = group.loc[group['type'] == 'offset']
            
            onset_rows = onset_rows.sort_values(by='pitch')
            offset_rows = offset_rows.sort_values(by='pitch')
            
            if last_offset_onset_value == 1:
                token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows())
                
                if len(offset_rows) > 0:
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(0))
                    last_offset_onset_value = 0
                    
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows())
            else:
                token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows())
                
                if len(onset_rows) > 0:
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(1))
                    last_offset_onset_value = 1
                    
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows())

        # ---------------------------- Add the end tokens ---------------------------- #
        
        # Add time shift to end if we haven't reached the end
        time_shift_to_end = np.floor(duration_in_ten_ms - cur_time)
        if time_shift_to_end != 0:
            token_sequence.append(self.vocabulary['time_shift'][0].translate_value_to_token(time_shift_to_end))
        
        # Add EOS 
        token_sequence.append(self.vocabulary['EOS'][0].translate_value_to_token(0))
                
        return token_sequence, df_tie_notes
    

class VocabBeat(Vocabulary):
    def __init__(self, config):
        super().__init__(config, dtype = "beat")
    
    def translate_sequence_events_to_tokens(self, duration, df_sequence: pd.DataFrame, df_tie_notes: pd.DataFrame,  song_name: str = 'Song_name_not_given') -> tuple[list[int], pd.DataFrame]:
        """See Transformer/docs/vocabulary.md for more info 
        
        Args:
            sequences (list[DataFrame]): Assumes for now a dataframe with pitch(int), onset(float), offset(float)
        """        
        subdivision = self.config["subdivision"]
        tuplet_subdivision = Fraction(self.config["tuplet_subdivision"])
        sub_tup_common_beats = len(np.intersect1d(np.arange(0, 1, subdivision), np.arange(0, 1, tuplet_subdivision)))
        
        # ------------------------------ Setup dataframe ----------------------------- #
        # Retrieve the initial tempo and remove from the dataframe
        tempo = -df_sequence.iloc[0]['pitch']
        if tempo <= 1:
            raise ValueError(f"Dataframe does not start with tempo. Got: {tempo}")
        df_sequence = df_sequence[df_sequence.index > 0]
        
        # Unravel the dataframe to have columns (pitch, type, time) where type is either onset or offset
        df_sequence['duration'] = df_sequence['offset'] - df_sequence['onset']
        df_sequence = df_sequence.melt(id_vars=['pitch', 'duration'], value_vars=['onset', 'offset'], var_name='type', value_name='time')
        
        # Compute tie notes where the offset is greater than the duration
        df_tie_note_offset_rows = ((df_sequence['type'] == 'offset') & (df_sequence['time'] > duration))
        df_tie_note_offsets = df_sequence[df_tie_note_offset_rows]
        df_tie_note_offsets['time'] -= duration # To get the offset relative to the duration of the next bar
        
        # Remove these rows from the sequence we want to tokenize
        df_sequence = df_sequence[~df_tie_note_offset_rows]
        
        # If any of the ET notes are being offset now, append them to the df_sequence
        # Also concat the ET notes that are not being offset now to the df_tie_note_offsets
        if df_tie_notes is not None:
            df_sequence = pd.concat([df_sequence, df_tie_notes[df_tie_notes['time'] <= duration]])
            
            df_tie_note_offsets = pd.concat([df_tie_note_offsets, df_tie_notes[df_tie_notes['time'] > duration]])
        
        # If there are no tie notes, set it to None
        if len(df_tie_note_offsets) == 0: 
            df_tie_note_offsets = None
        
        # Sort the sequence by time
        df_sequence = df_sequence.sort_values(by='time')
        
        # How many tuplets did we pass on the way (not including 1-beat tuplets)
        df_sequence['increment'] = df_sequence['time'] // tuplet_subdivision - np.floor(df_sequence['time'] / (1/sub_tup_common_beats))  
        # df_sequence['time'] = np.floor(df_sequence['time'] / subdivision) + df_sequence['increment']
        
        token_sequence = []
        # Add the start of sequence token
        token_sequence.append(self.vocabulary['SOS'][0].translate_value_to_token(0))
        token_sequence.append(self.vocabulary['tempo'][0].translate_value_to_token(tempo))
        
        # ----------------------------- Declare tie notes ----------------------------- #
        # NOTE: ET tokens happen before downbeats
        if df_tie_notes is not None:
            df_tie_notes = df_tie_notes.sort_values(by='pitch')
            
            for _, row in df_tie_notes.iterrows():
                token_sequence.append(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']))

            token_sequence.append(self.vocabulary['ET'][0].translate_value_to_token(0))
        
        # ------------------------ Get the rest of the labels ------------------------ #    
        
        cur_beat = 0 # NOTE: If we want beat 0 to be a token, remove this variable entirely since we always shift beats
        # First group by beats
        for beat_value, group in df_sequence.groupby('time'):
            # Number of subdivisions plus the off-beat tuplets we passed on the way
            beat_token = (np.floor(group['time'] / subdivision) + group['increment']).iloc[0]
            
            # Update which beat we are currently on
            if beat_value - cur_beat != 0: 
                token_sequence.append(self.vocabulary['beat'][0].translate_value_to_token(beat_token,song_name))
            
            # Update current time
            cur_beat = beat_value
            
            onset_rows = group.loc[group['type'] == 'onset']
            offset_rows = group.loc[group['type'] == 'offset']
            
            onset_rows = onset_rows.sort_values(by='pitch')
            offset_rows = offset_rows.sort_values(by='pitch')
            
            possible_grace_notes = onset_rows[(onset_rows['pitch'] >= 0) & (onset_rows['duration'] == 0)]
            
            # Look for tempo change
            if np.any(onset_rows['pitch'].values < -1):
                token_sequence.append(self.vocabulary['tempo'][0].translate_value_to_token(-onset_rows['pitch'].iloc[0]))
            # Look for downbeat
            if -1 in onset_rows['pitch'].values:
                token_sequence.append(self.vocabulary['downbeat'][0].translate_value_to_token(0))
                 
            # Start with offsetting before onsetting
            if len(offset_rows) > 0:    
                # Don't add offset for special tokens or grace notes (yet - preserve hierarchy)
                if np.any(offset_rows['duration'].values != 0):
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(0))
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows() if row['duration'] != 0)
                
            if len(onset_rows) > 0:
                       
                # Only add pitch tokens - a grace note is always accompanied by a note with a duration
                if np.any(onset_rows['duration'].values != 0):
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(1))
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows() if row['pitch'] >= 0)

                # Now add grace notes offsets (to preserve onset/offset convention)
                if not possible_grace_notes.empty:
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(0))
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in possible_grace_notes.iterrows())
            
            
        # ---------------------------- Add the end tokens ---------------------------- #
        
        # Add time shift to end if we haven't reached the end
        # time_shift_to_end = np.floor(duration - cur_beat)
        if cur_beat != duration:
            duration_token = np.floor(duration / subdivision) + (duration // tuplet_subdivision - np.floor(duration / (1/sub_tup_common_beats)))
            token_sequence.append(self.vocabulary['beat'][0].translate_value_to_token(duration_token,song_name))
        
        # Add EOS 
        token_sequence.append(self.vocabulary['EOS'][0].translate_value_to_token(0))
        
        # Return the token sequence and the notes that are offset later in the sequence        
        return token_sequence, df_tie_note_offsets
    

if __name__ == "__main__":
    import yaml

    with open('Transformer/configs/preprocess_config.yaml', 'r') as file:
        p_config = yaml.safe_load(file)
        
    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    if p_config['model'] == "TimeShift":
        vocab = VocabTime(config)
        vocab.define_vocabulary()
    elif p_config['model'] == "BeatTrack":
        vocab = VocabBeat(config)
        vocab.define_vocabulary(p_config['max_beats'])
    
    print("Vocabulary ranges: ")
    for token_range, (event, token_range) in vocab.vocabulary.items():
        print(f"Event: {event.event_type}, Range: {token_range}, Min-max: {event.min_max_values}")
    
    print('-'*40)
    
    print("Vocabulary size: ", vocab.vocab_size) 
    
    print('-'*40)
    # token_sequence = [1, 
    #                   133, 108-12*4, 382, 
    #                   132, 108-12*4, 1927, 
    #                   133, 22+12*2, 26+12*2, 491, 
    #                   132, 22+12*2, 
    #                   133, 37+12, 32+12, 234, 
    #                   132, 26+12*2, 37+12, 32+12, 13+12*3, 
    #                   2, 0]
    token_sequence = [423, 420, 410]
    
    print("Translated sequence: ")
    for translated_event in vocab.translate_sequence_token_to_events(token_sequence):
        print(f"Event: {translated_event[0]}, Value: {translated_event[1]}")