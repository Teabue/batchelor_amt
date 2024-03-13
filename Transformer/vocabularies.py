"""
Structure of the vocabulary:

The vocabulary containes several eventtypes
Keep a dictionairy of what ranges of tokens are what eventtype
The use that event.translate_from_token(token) to get the event value.
"""
import pandas as pd


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
        
        return token - self.token_start + self.min_max_values[0]
    
    def translate_value_to_token(self, value):
        if not value >= self.min_max_values[0] and value <= self.min_max_values[1]:
            raise ValueError('Value out of range')
        
        return value - self.min_max_values[0] + self.token_start
        
        
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
        self.vocabulary: dict[tuple[int,int], EventType] = {} 
        token_offset = 0
        
        # Set special tokens
        for special_token in self.config['special_tokens']:
            token_offset = self._define_event(token_offset, special_token, (0,0))

        for event_type, min_max_values  in self.config['event_types'].items():
            token_offset = self._define_event(token_offset, event_type, min_max_values)

        self.vocab_size = token_offset 

    def translate_sequence_events_to_tokens(self, duration, df_sequence: pd.DataFrame, df_tie_notes: pd.DataFrame, cur_time):
        """See Transformer/docs/vocabulary.md for more info 
        
        Args:
            sequences (list[DataFrame]): Assumes for now a dataframe with pitch(int), onset(float), offset(float)
        """        

        # ------------------------------ Setup dataframe ----------------------------- #
        # Unravel the dataframe to have columns (pitch, type, time) where type is either onset or offset
        df_sequence = df_sequence.melt(id_vars='pitch', value_vars=['onset', 'offset'], var_name='type', value_name='time')
        #thank god for copilot :^^DD
        
        df_sequence = df_sequence.sort_values(by='time')
        
        # TODO: MAKE THIS FLEXIBLE
        # Round to nearest 10 ms (0.01s) and convert to ms
        df_sequence['time'] = ((pd.to_numeric(df_sequence['time']) * 100).round() * 10)
        
        
        last_offset_onset_value = None
        token_sequence = []
        # -------------------------- Handle tie notes first -------------------------- #
        if df_tie_notes is not None:
            df_tie_notes = df_tie_notes.sort_values(by='time')
            
            for _, row in df_tie_notes.iterrows():
                token_sequence.append(self.vocabulary['pitch'].translate_value_to_token(row['pitch']))
  
            
            token_sequence.append(self.vocabulary['ET'].translate_value_to_token(0))
        
        # ------------------------ Get the rest of the labels ------------------------ #
        
        # First group by time
        for time_value, group in df_sequence.groupby('time'):
            if cur_time > duration:
                break
            
            if time_value > cur_time:
                token_sequence.append(self.vocabulary['time_shift'].translate_value_to_token(time_value - cur_time))
                cur_time = time_value
            
            # OH GAWD MY EYES ARE BURNING; REFACTOR THIS AT ALL COSTS AAAAaaa
            if last_offset_onset_value is None:
                token_sequence.append(self.vocabulary['offset_onset'].translate_value_to_token(1)) # start with onset
                last_offset_onset_value = 1
            
            onset_rows = group.loc[group['type'] == 'onset']
            offset_rows = group.loc[group['type'] == 'offset']
            
            onset_rows = onset_rows.sort_values(by='pitch')
            offset_rows = offset_rows.sort_values(by='pitch')
            
            if last_offset_onset_value == 1:
                token_sequence.extend(self.vocabulary['pitch'].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows())
                
                if len(offset_rows) > 0:
                    token_sequence.append(self.vocabulary['offset_onset'].translate_value_to_token(0))
                    last_offset_onset_value = 0
                    
                    token_sequence.extend(self.vocabulary['pitch'].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows())
            else:
                token_sequence.extend(self.vocabulary['pitch'].translate_value_to_token(row['pitch']) for _, row in offset_rows.iterrows())
                
                if len(onset_rows) > 0:
                    token_sequence.append(self.vocabulary['offset_onset'].translate_value_to_token(1))
                    last_offset_onset_value = 1
                    
                    token_sequence.extend(self.vocabulary['pitch'].translate_value_to_token(row['pitch']) for _, row in onset_rows.iterrows())

        # TODO: Add last time shift to get to the end
        # TODO; Add df_tie_notes
        
        return token_sequence, df_tie_notes
                
            
                

    def translate_sequence_token_to_events(self, sequence: list[int]):
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
    token_sequence = [4,10,1,130,4,10,131,9,10,12,130,9,10,12,0]
    
    print("Translated sequence: ")
    for translated_event in vocab.translate_sequence_token_to_events(token_sequence):
        print(f"Event: {translated_event[0]}, Value: {translated_event[1]}")