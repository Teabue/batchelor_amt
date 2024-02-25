class Vocabulary:
    def __init__(self):
        pass
    
    def define_vocabulary(self, vocab_config, extra_tokens):
        vocabulary = {} 
        token_offset = 0
        
        for type_event, min_max_values in vocab_config:
            value_width =  min_max_values[1] - min_max_values[0]
            
            vocabulary[type_event] = {'min-max': min_max_values, 'token_range': (token_offset, value_width + token_offset)}
            token_offset = value_width + 1
        
        for name in extra_tokens:
            vocabulary[name] = {'min-max': (1,1), 'token_range': (token_offset, token_offset)}
            token_offset += 2
            
        self.vocab_size = token_offset - 1 
        self.vocabulary = vocabulary
    
        
    



class MAPS_Vocabulary(Vocabulary):
    def tokenize():
        pass 
        