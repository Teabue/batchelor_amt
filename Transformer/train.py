import torch
import tqdm
import os
import yaml
import numpy as np
import torch.nn as nn
from utils.model import Transformer
from utils.data_loader import TransformerDataset  
from utils.vocabularies import Vocabulary



def train_model(model, criterion, optimizer, dataset: TransformerDataset, num_epochs=100, device='cuda', batch_size=20, nr_pitches=128, tgt_vocab_size=None):
    

    for epoch in range(num_epochs):
        
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(train_loader, total = len(train_loader), \
                         desc=f'Train: Loss: [{1}], Epochs: {epoch}/{num_epochs}', leave = False)    

        losses = []
        for spectrograms, tokens in pbar:#iter(train_loader):
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            
            output = model(src=spectrograms, tgt=tokens)
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tokens.contiguous().view(-1))
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            pbar.set_description(f'Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            losses.append(loss.item())
        
        print(f'Train res; Epoch: {epoch}, Loss: {sum(losses)/len(losses)}')

        test_loader = dataset.get_split('val', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(iter(test_loader), total = len(test_loader), \
                        desc=f'Loss: [{1}], Epochs: {epoch}/{num_epochs}')
        
        best_loss = float('inf')
        losses = []
        correct = 0
        for i, (spectrograms, tokens) in enumerate(pbar):
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            if i > 200:
                break
                
            outputs = model(src=spectrograms, tgt=tokens)
            
            loss = criterion(outputs.contiguous().view(-1, tgt_vocab_size), tokens.contiguous().view(-1))
            
            pbar.set_description(f'Validation: Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            losses.append(loss.item())
            
        # Save model if loss is better
        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            MODEL_SAVE_DIR = os.path.join('/work3/s214629', "models")
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'model_best.pth'))  


def simple_setup(lr = 0.01, device = 'cuda', nr_pitches=128, datadir = 'data_process/output/segments'):
    #TODO: Make this into configs
    n_mel_bins = 128
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 1024 # doesn't really matter, can be set to whatever as long as its larger than the longest sequence - it's a pteprocess step for PE
    dropout = 0.1
    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    vocab = Vocabulary(config)
    vocab.define_vocabulary()
    
    tgt_vocab_size = vocab.vocab_size
    
    model = Transformer(n_mel_bins, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device)
    model.train()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TransformerDataset(datadir)
    
    return model, criterion, optimizer, dataset, tgt_vocab_size


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>>>> Using device: ", DEVICE)
    model, criterion, optimizer, dataset, tgt_vocab_size = simple_setup(device = DEVICE, datadir = '/work3/s214629/preprocessed_data')
    
    train_model(model, criterion, optimizer, dataset, num_epochs=20, device=DEVICE, batch_size=150, tgt_vocab_size=tgt_vocab_size)