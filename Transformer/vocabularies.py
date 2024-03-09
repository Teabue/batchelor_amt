"""
Structure of the vocabulary:

The vocabulary containes several eventtypes
Keep a dictionairy of what ranges of tokens are what eventtype
The use that event.translate_from_token(token) to get the event value.
"""


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
    
    def translate_from_token(self, token):
        if not token >= self.token_start and token <= self.end_token:
            raise ValueError('Token out of range')
        
        return token - self.token_start + self.min_max_values[0]
        
        
class Vocabulary:
    def __init__(self, config):
        self.config = config
    
    
    def define_vocabulary(self):
        self.vocabulary: dict[tuple[int,int], EventType] = {} 
        token_offset = 0
        
        # Set special tokens
        for special_token in self.config['special_tokens']:
            min_token = token_offset
            event = EventType(special_token, (0,0))
            token_offset = event.set_token_range(token_offset)
            
            self.vocabulary[(min_token, token_offset - 1)] = event
        
        for event_types, min_max_values  in self.config['event_types'].items():
            min_token = token_offset
            event = EventType(event_types, min_max_values)
            token_offset = event.set_token_range(token_offset)
            
            self.vocabulary[(min_token, token_offset - 1)] = event
        

        self.vocab_size = token_offset 

        
    def translate_sequence(self, sequence: list[int]):
        """Translates a sequence of tokens to a sequence of events
        Used only in inference to create midi 
        Args:
            sequence (list[int])
        """
        # NOTE: This is probably a really slow way to do it, alternative implementation up for consideration
        # The input will even prolly be a tensor :^)
        
        translated_sequence = []
        for token in sequence:
            
            for token_range, event in self.vocabulary.items():
                if token >= token_range[0] and token <= token_range[1]:
                    event_value = event.translate_from_token(token)
                    translated_sequence.append((event.event_type, event_value))
                    break
    


    