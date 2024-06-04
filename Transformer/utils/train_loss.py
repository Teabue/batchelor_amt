import torch
from utils.vocabularies import Vocabulary
# from vocabularies import Vocabulary
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, vocabulary: Vocabulary, small_eps = 1e-10, ignore_class: int = None, device = None):
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
                

    def forward(self, logits, ground_truths):
        return self.compute_loss(logits, ground_truths)
    
    def compute_loss(self, pitch_logits, labels) -> torch.Tensor:
        loss = 0
        # Pre-compute some values
        exp_pitch_logits = torch.exp(pitch_logits)
        sum_exp_pitch_logits = torch.sum(exp_pitch_logits, dim=1, keepdim=True) + self.small_eps
        max_logit_indices = torch.argmax(pitch_logits, dim=1)

        # Initialize masks and distances
        mask1 = torch.zeros((len(labels), self.vocab.vocab_size)).to(self.device)
        mask2 = torch.ones((len(labels), self.vocab.vocab_size)).to(self.device)

        # Compute masks and distances
        for k in range(len(labels)):
            predicted_event_type = self.vocab.translate_sequence_token_to_events([max_logit_indices[k]])[0][0]
            ground_truth_event_type = self.vocab.translate_sequence_token_to_events([labels[k]])[0][0]
            token_range_real_label = self.vocab.vocabulary[ground_truth_event_type][1]
            
            
            if predicted_event_type == ground_truth_event_type:        
                mask1[k, token_range_real_label[0]:token_range_real_label[1]] = 1
                mask2[k, :] = 0
            else:
                mask2[k, token_range_real_label[0]:token_range_real_label[1]] = 0
            
            
        # Compute distances
        distances = torch.abs(max_logit_indices.unsqueeze(1) - labels.unsqueeze(1)) * exp_pitch_logits / sum_exp_pitch_logits * mask1 / (torch.sum(mask1, dim=1, keepdim=True) + self.small_eps) + mask2 / (torch.sum(mask2, dim=1, keepdim=True) + self.small_eps) * (1 + exp_pitch_logits / sum_exp_pitch_logits) * self.upper_bound
        
        # Create ignore mask
        if self.ignore_class != None:
            ignore_mask = (labels != self.ignore_class).float().unsqueeze(1).to(self.device)
            distances = distances * ignore_mask

        # Compute loss
        loss = torch.sum(distances)
        
        return loss
    

if __name__ == '__main__':
    import yaml 

    with open('/zhome/5d/a/168095/batchelor_amt/Transformer/configs/vocab_config.yaml', 'r') as f:
        vocab_config = yaml.safe_load(f)


    vocab = Vocabulary(vocab_config)
    vocab.define_vocabulary(4)

    pitch_logits = torch.randn(10, vocab.vocab_size)
    labels = torch.randint(0, vocab.vocab_size, (10,))
    
    loss = CustomLoss(vocab)
    
    loss.compute_loss(pitch_logits, labels)
    
    
    
    
    # ----------------------------------- trash ---------------------------------- #
    
#     mask1_test1 = torch.zeros((len(labels), vocab.vocab_size))
# mask2_test1 = torch.ones((len(labels), vocab.vocab_size))
# loss = 0
# for k in range(len(labels)):
#     for i in range(vocab.vocab_size):
        
#         max_logit_index = torch.argmax(pitch_logits[k])
        
#         predicted_event_type = vocab.translate_sequence_token_to_events([max_logit_index])[0][0]
#         ground_truth_event_type = vocab.translate_sequence_token_to_events([labels[k]])[0][0]
#         token_range_real_label = vocab.vocabulary[ground_truth_event_type][1]
#         if predicted_event_type == ground_truth_event_type:        
#             mask1 = torch.zeros(vocab.vocab_size)
#             mask1[token_range_real_label[0]:token_range_real_label[1]] = 1
#             mask2 = torch.zeros(vocab.vocab_size)
            

#         else:
#             mask1 = torch.zeros(vocab.vocab_size)
#             mask2 = torch.ones(vocab.vocab_size)
#             mask2[token_range_real_label[0]:token_range_real_label[1]] = 0
            
        
#         # if k == 0 and i == 0:
#         #     print(labels[k], max_logit_index)
#         #     print('mask1', mask1)
#         #     print('mask2', mask2)
#         #     print('predicted_event_type', predicted_event_type, 'ground_truth_event_type', ground_truth_event_type)
        
#         mask1_test1[k] = mask1
#         mask2_test1[k] = mask2 
#         distance = torch.abs(max_logit_index - labels[k]) * torch.exp(pitch_logits[k][i])/torch.sum(torch.exp(pitch_logits[k][i])) * mask1[i]/(torch.sum(mask1) + small_eps) + mask2[i]/(torch.sum(mask2) + small_eps) * (1 + torch.exp(pitch_logits[k][i])/torch.sum(torch.exp(pitch_logits[k][i]))) * upper_bound
        
#         loss += distance