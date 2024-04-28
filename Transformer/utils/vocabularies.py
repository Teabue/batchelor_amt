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
        if not token >= self.token_start and token <= self.end_token:
            raise ValueError('Token out of range')
        
        return int(token - self.token_start + self.min_max_values[0]) # NOTE: Assume only integer values for now
    
    def translate_value_to_token(self, value):
        if not value >= self.min_max_values[0] and value <= self.min_max_values[1]:
            raise ValueError('Vocabulary: Value out of range')
        
        return int(value - self.min_max_values[0] + self.token_start)
        
        
class Vocabulary:
    def __init__(self, config):
        self.config = config
    
    
    def _define_event(self, token_offset, token_type, min_max_values):
        min_token = token_offset
        event = EventType(token_type, min_max_values)
        token_offset = event.set_token_range(token_offset)
        self.vocabulary[token_type] = (event, (min_token, token_offset - 1))
        return token_offset
    
    
    def define_vocabulary(self):
        self.vocabulary: dict[EventType, tuple[int,int]] = {} 
        token_offset = 0
        
        # Set special tokens
        for special_token in self.config['special_tokens']:
            token_offset = self._define_event(token_offset, special_token, (0,0))

        for event_type, min_max_values  in self.config['event_types'].items():
            token_offset = self._define_event(token_offset, event_type, min_max_values)

        self.vocab_size = token_offset 

    def translate_sequence_events_to_tokens(self, duration, df_sequence: pd.DataFrame, df_tie_notes: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
        """See Transformer/docs/vocabulary.md for more info 
        
        Args:
            sequences (list[DataFrame]): Assumes for now a dataframe with pitch(int), onset(float), offset(float)
        """        
        subdivision = self.config["subdivision"]
        tuplet_subdivision = Fraction(self.config["tuplet_subdivision"])
        
        # ---------------- In case no notes are played in the sequence --------------- #
        if len(df_sequence) == 0:
            duration_token = np.floor(duration / subdivision) + (duration // tuplet_subdivision - np.floor(duration))
            token_sequence = []
            token_sequence.append(self.vocabulary['SOS'][0].translate_value_to_token(0))
            token_sequence.append(self.vocabulary['beat'][0].translate_value_to_token(duration_token))
            token_sequence.append(self.vocabulary['EOS'][0].translate_value_to_token(0))
            return token_sequence, None
        
        # ------------------------------ Setup dataframe ----------------------------- #
        # Unravel the dataframe to have columns (pitch, type, time) where type is either onset or offset
        df_sequence = df_sequence.melt(id_vars='pitch', value_vars=['onset', 'offset'], var_name='type', value_name='time')
        #thank god for copilot :^^DD
        
        # Compute tie notes where the offset is greater than the duration
        df_tie_note_offset_rows = ((df_sequence['type'] == 'offset') & (df_sequence['time'] > duration))
        df_tie_note_offsets = df_sequence[df_tie_note_offset_rows]
        
        # Remove these rows from the sequence we want to tokenize
        df_sequence = df_sequence[~df_tie_note_offset_rows]
        
        # If any of the ET notes are being offset now, append them to the df_sequence
        # Also concat the ET notes that are not being offset now to the df_tie_note_offsets
        if df_tie_notes is not None:
            df_tie_notes['time'] -= duration
            df_sequence = pd.concat([df_sequence, df_tie_notes[df_tie_notes['time'] <= duration]])
            
            df_tie_note_offsets = pd.concat([df_tie_note_offsets, df_tie_notes[df_tie_notes['time'] > duration]])
        
        if len(df_tie_note_offsets) == 0: # If there are no tie notes, set it to None
            df_tie_note_offsets = None
        
        # Sort the sequence by time
        df_sequence = df_sequence.sort_values(by='time')
        
        # How many tuplets did we pass on the way (not including 1-beat tuplets)
        sub_tup_common_beats = len(np.intersect1d(np.arange(0, 1, subdivision), np.arange(0, 1, tuplet_subdivision)))
        df_sequence['increment'] = df_sequence['time'] // tuplet_subdivision - np.floor(df_sequence['time'] / (1/sub_tup_common_beats))  
        # df_sequence['time'] = np.floor(df_sequence['time'] / subdivision) + df_sequence['increment']
        
        token_sequence = []
        # Add the start of sequence token
        token_sequence.append(self.vocabulary['SOS'][0].translate_value_to_token(0))

        # ----------------------------- Declare tie notes ----------------------------- #
        # NOTE: ET tokens happen before downbeats
        if df_tie_notes is not None:
            df_tie_notes = df_tie_notes.sort_values(by='pitch')
            
            for _, row in df_tie_notes.iterrows():
                token_sequence.append(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']))

            token_sequence.append(self.vocabulary['ET'][0].translate_value_to_token(0))
        
        # ------------------------ Get the rest of the labels ------------------------ #    
           
        cur_beat = 0
        # First group by beats
        for beat_value, group in df_sequence.groupby('time'):
            # Number of subdivisions plus the off-beat tuplets we passed on the way
            beat_token = (np.floor(group['time'] / subdivision) + group['increment']).iloc[0]
            
            # Update which beat we are currently on
            if beat_value - cur_beat != 0: 
                token_sequence.append(self.vocabulary['beat'][0].translate_value_to_token(beat_token))
            
            # Update current time
            cur_beat = beat_value
            
            onset_rows = group.loc[group['type'] == 'onset']
            offset_rows = group.loc[group['type'] == 'offset']
            
            onset_rows = onset_rows.sort_values(by='pitch')
            offset_rows = offset_rows.sort_values(by='pitch')
            
            # Start with offsetting before onsetting
            if len(offset_rows) > 0:
                # Don't add tokens if it's just a downbeat and no notes
                if not (len(offset_rows) == 1 and offset_rows['pitch'].values[0] == -1):
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(0))
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows() if row['pitch'] != -1)
                
            if len(onset_rows) > 0:
                if -1 in onset_rows['pitch'].values:
                    token_sequence.append(self.vocabulary['downbeat'][0].translate_value_to_token(0))
                        
                # Don't add tokens if it's just a downbeat and no notes
                if not (len(onset_rows) == 1 and onset_rows['pitch'].values[0] == -1):
                    token_sequence.append(self.vocabulary['offset_onset'][0].translate_value_to_token(1))
                    token_sequence.extend(self.vocabulary['pitch'][0].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows() if row['pitch'] != -1)
                
        # ---------------------------- Add the end tokens ---------------------------- #
        
        # Add time shift to end if we haven't reached the end
        # time_shift_to_end = np.floor(duration - cur_beat)
        if cur_beat != duration:
            duration_token = np.floor(duration / subdivision) + (duration // tuplet_subdivision - np.floor(duration))
            token_sequence.append(self.vocabulary['beat'][0].translate_value_to_token(duration_token))
        
        # Add EOS 
        token_sequence.append(self.vocabulary['EOS'][0].translate_value_to_token(0))
        
        # Return the token sequence and the notes that are offset later in the sequence        
        return token_sequence, df_tie_note_offsets
                
                
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
    

if __name__ == "__main__":
    import yaml

    
    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    vocab = Vocabulary(config)
    
    vocab.define_vocabulary()
    
    print("Vocabulary ranges: ")
    for token_range, (event, token_range) in vocab.vocabulary.items():
        print(f"Event: {event.event_type}, Range: {token_range}, Min-max: {event.min_max_values}")
    
    print('-'*40)
    token_sequence = [1, 
                      133, 108-12*4, 382, 
                      132, 108-12*4, 1927, 
                      133, 22+12*2, 26+12*2, 491, 
                      132, 22+12*2, 
                      133, 37+12, 32+12, 234, 
                      132, 26+12*2, 37+12, 32+12, 13+12*3, 
                      2, 0]
    
    print("Translated sequence: ")
    for translated_event in vocab.translate_sequence_token_to_events(token_sequence):
        print(f"Event: {translated_event[0]}, Value: {translated_event[1]}")