import argparse
import torch
import tqdm
import os
import yaml
import numpy as np
import torch.nn as nn
from utils.model import Transformer
from utils.data_loader_2 import TransformerDataset  
from utils.vocabularies import VocabBeat, VocabTime
from utils.train_loss import CustomLoss


def train_model(model, 
                criterion, 
                optimizer, 
                dataset: TransformerDataset, 
                num_epochs=100, 
                device='cuda', 
                batch_size=20, 
                tgt_vocab_size=None, 
                run_save_path='', 
                save_model_every_num_epoch=1):
    
    
    MODEL_SAVE_DIR = os.path.join(run_save_path, 'models')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    LOSS_SAVE_DIR = os.path.join(run_save_path, 'losses')
    os.makedirs(LOSS_SAVE_DIR, exist_ok=True)
    
    TRAIN_LOSS_PATH = os.path.join(LOSS_SAVE_DIR, 'train_losses.txt')
    VAL_LOSS_PATH = os.path.join(LOSS_SAVE_DIR, 'val_losses.txt')
    TEST_LOSS_PATH = os.path.join(LOSS_SAVE_DIR, 'test_losses.txt')
    
    with open(TRAIN_LOSS_PATH, 'w') as f:
        f.write('Epoch, Loss\n')
    with open(VAL_LOSS_PATH, 'w') as f:
        f.write('Epoch, Loss\n')
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        
        # -------------------------------- Train model ------------------------------- #
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(train_loader, total = len(train_loader), \
                         desc=f'Train: Loss: [{1}], Epochs: {epoch}/{num_epochs}, GPU Memory: {0}GB', leave = False)    
        model.train()
        train_losses = []
        for spectrograms, tokens in pbar:
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            
            output = model(src=spectrograms, tgt=tokens[:, :-1])
            output = output[:, 1:, :] # Remove the prediction of the second token because we give it two tokens right off
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tokens[:, 2:].contiguous().view(-1)) # tokens[:, 2:] - Don't account loss for neither SOS or declared tempo
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
                        
            if device.type == 'cuda':
                pbar.set_description(f'Train Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Train Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
                
            train_losses.append(loss.item())
            
            # Me trying to free memory because for some reason, the memory usage keeps increasing
            del output, loss, spectrograms, tokens
        
        mean_train_loss = np.mean(train_losses)
        print(f'Train res; Epoch: {epoch}, Loss: {mean_train_loss}')
        
        # Save training loss
        with open(TRAIN_LOSS_PATH, 'a') as f:
            f.write(f'{epoch}, {mean_train_loss}\n')
            
        if epoch % save_model_every_num_epoch == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'checkpoint_model.pth'))
        
        
        # ------------------------------ Validate model ------------------------------ #
        val_loader = dataset.get_split('val', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(iter(val_loader), total = len(val_loader), \
                        desc=f'Loss: [{1}], Epochs: {epoch}/{num_epochs}')
        
        model.eval()
        
        val_losses = []
        for i, (spectrograms, tokens) in enumerate(pbar):
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            if i > 200:
                break
                
            outputs = model(src=spectrograms, tgt=tokens[:, :-1])
            
            loss = criterion(outputs.contiguous().view(-1, tgt_vocab_size), tokens[:, 1:].contiguous().view(-1))
            if device.type == 'cuda':
                pbar.set_description(f'Validation Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Validation Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            val_losses.append(loss.item())
            
            # Don't mind me trying to free memory
            del outputs, loss, spectrograms, tokens
        
        # Save model if loss is better
        mean_val_loss = np.mean(val_losses)
        
        # Save validation loss
        with open(VAL_LOSS_PATH, 'a') as f:
            f.write(f'{epoch}, {mean_val_loss}\n')
        

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'model_best.pth'))  
            
            
        # --------------------------------- Test loss -------------------------------- #

        test_loader = dataset.get_split('test', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(iter(test_loader), total = len(test_loader), \
                        desc=f'Loss: [{1}], Epochs: {epoch}/{num_epochs}')
        
        model.eval()
        
        test_losses = []
        for i, (spectrograms, tokens) in enumerate(pbar):
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            if i > 200:
                break
                
            outputs = model(src=spectrograms, tgt=tokens[:, :-1])
            
            loss = criterion(outputs.contiguous().view(-1, tgt_vocab_size), tokens[:, 1:].contiguous().view(-1))
            if device.type == 'cuda':
                pbar.set_description(f'Test Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Test Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            test_losses.append(loss.item())
            
            # Don't mind me trying to free memory
            del outputs, loss, spectrograms, tokens
        
        # Save model if loss is better
        mean_test_loss = np.mean(test_losses)
        
        # Save validation loss
        with open(TEST_LOSS_PATH, 'a') as f:
            f.write(f'{epoch}, {mean_test_loss}\n')


def simple_setup(device = 'cuda', 
                data_dir = 'data_process/output/segments',
                lr = 0.0001,  
                n_mel_bins = 128, 
                d_model = 512,
                num_heads = 8,
                num_layers = 6,
                d_ff = 2048,
                max_seq_length = 1024, # doesn't really matter, can be set to whatever as long as its larger than the longest sequence - it's a pteprocess step for PE
                dropout = 0.1,
                pretrained_run_path = None,
                data_parallelism=False):

    # TODO: Temporary solution, way into the future when I actually feel like refactoring, I'll save config run files
    with open(os.path.join(data_dir,'preprocess_config.yaml'), 'r') as file:
        p_config = yaml.safe_load(file)
    
    with open(os.path.join(data_dir,'vocab_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    
    if p_config['model'] == "TimeShift":
        vocab = VocabTime(config)
        vocab.define_vocabulary()
    elif p_config['model'] == "BeatTrack":
        vocab = VocabBeat(config)
        vocab.define_vocabulary(p_config['max_beats'])
    else:
        raise ValueError('Model type not recognized')
    
    tgt_vocab_size = vocab.vocab_size
    
    model = Transformer(n_mel_bins, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device)
    if pretrained_run_path != None:
        model.load_state_dict(torch.load(os.path.join(pretrained_run_path,'models','model_best.pth')))
    
    if data_parallelism:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    dataset = TransformerDataset(data_dir)
    
    return model, vocab, optimizer, dataset, tgt_vocab_size


if __name__ == '__main__':
    ''' Run the file from the repo root folder'''
    import yaml
    
    parser = argparse.ArgumentParser(description='Train a model with specified loss function.')
    parser.add_argument('--loss', type=str, choices=['ce', 'cl'], help="Specify the loss function", default='ce')
    args = parser.parse_args()
    
    pretrained_run_path = None
    data_parallelism = False
    
    # SET THIS, I'M TOO LAZY TO ARGUMENT PARSE
    # pretrained_run_path = '/work3/s214629/run_a100_hope3_cont_smallest_lr' # Comment out to train a new model from scratch

    if pretrained_run_path != None:
        with open(os.path.join(pretrained_run_path,'train_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        # CHANGE NEEDED SETTINGS HERE
        config['run_save_path'] = '/work3/s214629/runs/05-06-24_time_shift_gen-audio_aug_cont_00001'
        config['num_epochs'] = 69
        config['seed'] = config['seed'] + 33 # We need to set to something else so the batches won't be identical
        config['lr'] = 0.00001
        if data_parallelism:
            config['batch_size'] =  40
    else:
        # Load the YAML file
        with open('Transformer/configs/train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    config['run_save_path'] = os.path.join(config['run_base_path'], config['run_specific_path'], args.loss)
    
    # Save the used configs to the run_save_path
    os.makedirs(config['run_save_path'], exist_ok=True)
    
    with open(os.path.join(config['run_save_path'], 'train_config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
        
    torch.manual_seed(config['seed'])
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('>>>>> Using device: ', DEVICE)
    
    model, vocab, optimizer, dataset, tgt_vocab_size = simple_setup(device = DEVICE, 
                                                                        data_dir = config['data_dir'], 
                                                                        lr = config['lr'], 
                                                                        n_mel_bins = config['n_mel_bins'], 
                                                                        d_model = config['d_model'], 
                                                                        num_heads = config['num_heads'], 
                                                                        num_layers = config['num_layers'], 
                                                                        d_ff = config['d_ff'], 
                                                                        max_seq_length = config['max_seq_length'], 
                                                                        dropout = config['dropout'],
                                                                        pretrained_run_path=pretrained_run_path,
                                                                        data_parallelism=data_parallelism)
    
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    elif args.loss == 'cl':
        criterion = CustomLoss(vocab).compute_loss
    else:
        raise ValueError('Loss function not recognized')
    
    train_model(model, 
                criterion, 
                optimizer, 
                dataset, 
                num_epochs=config['num_epochs'], 
                device=DEVICE, 
                batch_size=config['batch_size'], 
                tgt_vocab_size=tgt_vocab_size,
                run_save_path=config['run_save_path'])