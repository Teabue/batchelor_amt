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
        pass
    
    def define_vocabulary(self, vocab_config, extra_tokens):
        vocabulary = {} 
        token_start = 0 # Keep it as 0 for now
        
        for type_event, min_max_values in vocab_config:
            value_width =  min_max_values[1] - min_max_values[0]
            
            vocabulary[type_event] = {'min-max': min_max_values, 'token_range': (token_offset, value_width + token_offset)}
            token_offset = value_width + 1
        
        for name in extra_tokens:
            vocabulary[name] = {'min-max': (1,1), 'token_range': (token_offset, token_offset)}
            token_offset += 2
            
        self.vocab_size = token_offset - 1 
        self.vocabulary = vocabulary
        
    def translate_sequence tils saif 
    


    