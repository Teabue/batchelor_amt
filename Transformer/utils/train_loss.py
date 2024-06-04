import torch
from utils.vocabularies import Vocabulary
# from vocabularies import Vocabulary
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

class CustomLoss(nn.Module):
    def __init__(self, vocabulary: Vocabulary, small_eps = 1e-10, ignore_class: int = None, device = None, logger_folder = None):
        super(CustomLoss, self).__init__()
        self.vocab = vocabulary
        self.small_eps = small_eps
        self.ignore_class = ignore_class
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        upper_bound = float('-inf')
        for event_name, (event_type, token_range) in self.vocab.vocabulary.items():
            token_distance = (token_range[1] - token_range[0])
            if upper_bound < token_distance:
                self.upper_bound = token_distance
                
        if logger_folder:
            self.initialize_logger(logger_folder)
                
    def initialize_logger(self, logger_folder):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

        # Create a file handler
        handler = logging.FileHandler(os.path.join(logger_folder,'train_loss.log'))
        handler.setLevel(logging.WARNING)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(handler)
    def forward(self, logits, ground_truths):
        return self.compute_loss(logits, ground_truths)
    
    def compute_loss3(self, pitch_logits, labels) -> torch.Tensor:
        loss = 0
        
        true_event_types  = self.vocab.translate_sequence_token_to_events(labels)
        
        true_token_ranges = []
        upper_bounds = []
        for true_event_type in true_event_types:
            true_token_range = self.vocab.vocabulary[true_event_type[0]][1]
            true_token_ranges.append(true_token_range)
            
            upper_bounds.append(true_token_range[1] - true_token_range[0] + 1)
        
        for k in range(len(labels)):
            loss_reg = 0
            loss_class = 0
            
            # Regression loss
            for i in range(true_token_ranges[k][0], true_token_ranges[k][1] + 1):
                loss_reg += torch.abs(labels[k] - i + self.small_eps) * torch.exp(pitch_logits[k][i])/torch.sum(torch.exp(pitch_logits[k]))

            # Class loss
            for j in range(0, self.vocab.vocab_size):
                if j in range(true_token_ranges[k][0], true_token_ranges[k][1] + 1):
                    continue
                loss_class += torch.exp(pitch_logits[k][j])/torch.sum(torch.exp(pitch_logits[k])) * upper_bounds[k]
                
            loss_k = (loss_reg + loss_class) / upper_bounds[k]
            
            assert 0 <  loss_reg < (upper_bounds[k] - 1 + self.small_eps) 
            assert 0 <= loss_class <= upper_bounds[k]
            assert 0 < loss_class + loss_reg <= upper_bounds[k]
            assert 0 < loss_k <= 1
            
            loss += loss_k
        
        loss = loss / len(labels)        
        return loss
    
    def compute_loss2(self, pitch_logits, labels):
        true_event_types  = self.vocab.translate_sequence_token_to_events(labels)
                
        true_token_ranges = [self.vocab.vocabulary[true_event_type[0]][1] for true_event_type in true_event_types]
        upper_bounds = [(true_token_range[1] - true_token_range[0] + 1) for true_token_range in true_token_ranges]

        loss_reg = torch.zeros(len(labels))
        loss_class = torch.zeros(len(labels))

        # Regression loss
        for k in range(len(labels)):
            i = torch.arange(true_token_ranges[k][0], true_token_ranges[k][1] + 1)
            loss_reg[k] = torch.sum(torch.abs(labels[k] - i + self.small_eps) * torch.exp(pitch_logits[k][i])/torch.sum(torch.exp(pitch_logits[k])))

        # Class loss
        for k in range(len(labels)):
            j = torch.arange(0, self.vocab.vocab_size)
            mask = (j < true_token_ranges[k][0]) | (j > true_token_ranges[k][1])
            j = j[mask]
            loss_class[k] = torch.sum(torch.exp(pitch_logits[k][j])/torch.sum(torch.exp(pitch_logits[k])) * upper_bounds[k])

        loss_k = (loss_reg + loss_class) / torch.tensor(upper_bounds)

        assert torch.all((0 <  loss_reg) & (loss_reg < torch.tensor(upper_bounds) - 1 + self.small_eps))
        assert torch.all((0 <= loss_class) & (loss_class <= torch.tensor(upper_bounds)))
        assert torch.all((0 < (loss_class + loss_reg)) & ((loss_class + loss_reg) <= torch.tensor(upper_bounds)))
        assert torch.all((0 < loss_k) & (loss_k <= 1))

        loss = torch.mean(loss_k)

        return loss
    
    def compute_loss(self, pitch_logits, labels):
        true_event_types  = self.vocab.translate_sequence_token_to_events(labels)
        true_token_ranges = torch.tensor([self.vocab.vocabulary[true_event_type[0]][1] for true_event_type in true_event_types]).to(self.device)
        upper_bounds = torch.tensor([(true_token_range[1] - true_token_range[0] + 1) for true_token_range in true_token_ranges]).to(self.device)

        # Create a tensor of size (len(labels), self.vocab.vocab_size)
        k = torch.arange(len(labels), device=self.device).unsqueeze(1).expand(-1, self.vocab.vocab_size)
        i = torch.arange(self.vocab.vocab_size, device=self.device).expand(len(labels), -1)

        # Create masks for regression and class loss
        mask_reg = (i >= true_token_ranges[k, 0]) & (i <= true_token_ranges[k, 1])
        mask_class = (i < true_token_ranges[k, 0]) | (i > true_token_ranges[k, 1])

        # Compute softmax of pitch_logits
        softmax_pitch_logits = torch.softmax(pitch_logits, dim=1)

        # Regression loss
        loss_reg = torch.sum(mask_reg * torch.abs(labels.unsqueeze(1).expand_as(i) - i + self.small_eps) * softmax_pitch_logits, dim=1)

        # Class loss
        loss_class = torch.sum(torch.mul(upper_bounds.unsqueeze(1), mask_class * softmax_pitch_logits), dim=1)

        loss_k = (loss_reg + loss_class) / upper_bounds

        if self.logger == None:
            assert torch.all((0 <  loss_reg) & (loss_reg < upper_bounds - 1 + self.small_eps))
            assert torch.all((0 <= loss_class) & (loss_class <= upper_bounds))
            assert torch.all((0 < (loss_class + loss_reg)) & ((loss_class + loss_reg) <= upper_bounds))
            assert torch.all((0 < loss_k) & (loss_k <= 1))
        else:
            if not torch.all((0 <  loss_reg) & (loss_reg < upper_bounds - 1 + self.small_eps)):
                self.logger.warning('Assertion failed: 0 < loss_reg < upper_bounds - 1 + small_eps')
            if not torch.all((0 <= loss_class) & (loss_class <= upper_bounds)):
                self.logger.warning('Assertion failed: 0 <= loss_class <= upper_bounds')
            if not torch.all((0 < (loss_class + loss_reg)) & ((loss_class + loss_reg) <= upper_bounds)):
                self.logger.warning('Assertion failed: 0 < (loss_class + loss_reg) <= upper_bounds')
            if not torch.all((0 < loss_k) & (loss_k <= 1)):
                self.logger.warning('Assertion failed: 0 < loss_k <= 1')

        loss = torch.mean(loss_k)

        return loss

if __name__ == '__main__':
    import yaml 

    with open('/zhome/5d/a/168095/batchelor_amt/Transformer/configs/vocab_config.yaml', 'r') as f:
        vocab_config = yaml.safe_load(f)


    vocab = Vocabulary(vocab_config)
    vocab.define_vocabulary(4)
    num_test_tokens = 2000
    pitch_logits = torch.randn(num_test_tokens, vocab.vocab_size)
    labels = torch.randint(0, vocab.vocab_size, (num_test_tokens,))
    
    loss_class = CustomLoss(vocab)
    
    loss = loss_class.compute_loss(pitch_logits, labels)
    
    print(loss)
    
    loss2 = loss_class.compute_loss2(pitch_logits, labels)
    
    print(loss2)
    
    loss3 = loss_class.compute_loss3(pitch_logits, labels)
    print(loss3)