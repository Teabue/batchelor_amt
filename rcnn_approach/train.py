import torch
import tqdm
import os
import numpy as np
import torch.nn as nn

from dataloader import Dataset
from model import Resnext50

def train_model(model, criterion, optimizer, dataset: Dataset, run_save_path, num_epochs=100, device='cuda', batch_size=20, nr_pitches=128):

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
    with open(TEST_LOSS_PATH, 'w') as f:
        f.write('Epoch, Loss\n')
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        
        # -------------------------------- Train model ------------------------------- #
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(train_loader, total = len(train_loader), \
                         desc=f'Train: Loss: [{1}], Epochs: {epoch}/{num_epochs}', leave = False)    

        model.train()
        train_losses = []
        for images, labels in pbar:#iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
                        
            if device.type == 'cuda':
                pbar.set_description(f'Train Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Train Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
                
            train_losses.append(loss.item())
            
            # Me trying to free memory because for some reason, the memory usage keeps increasing
            del outputs, loss, images, labels
        
        mean_train_loss = np.mean(train_losses)
        print(f'Train res; Epoch: {epoch}, Loss: {mean_train_loss}')
        
        # Save training loss
        with open(TRAIN_LOSS_PATH, 'a') as f:
            f.write(f'{epoch}, {mean_train_loss}\n')
            
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'checkpoint_model.pth'))
        
        
        # ------------------------------ Validate model ------------------------------ #
        val_loader = dataset.get_split('val', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(iter(val_loader), total = len(val_loader), \
                        desc=f'Loss: [{1}], Epochs: {epoch}/{num_epochs}')
        
        model.eval()
        
        val_losses = []
        for images, labels in pbar:#iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)  
                      
            if device.type == 'cuda':
                pbar.set_description(f'Validation Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Validation Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            val_losses.append(loss.item())
            
            # Don't mind me trying to free memory
            del outputs, loss, images, labels
        
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
        for images, labels in pbar:#iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            if device.type == 'cuda':
                pbar.set_description(f'Test Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}, GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
            else:
                pbar.set_description(f'Test Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            test_losses.append(loss.item())
            
            # Don't mind me trying to free memory
            del outputs, loss, images, labels
        
        # Save model if loss is better
        mean_test_loss = np.mean(test_losses)
        
        # Save validation loss
        with open(TEST_LOSS_PATH, 'a') as f:
            f.write(f'{epoch}, {mean_test_loss}\n')
            
            
        


def simple_setup(lr = 0.001, wd = 0.0001, device = 'cuda', nr_pitches=128, datadir = 'data_process/output/segments'):
    
    model = Resnext50(nr_pitches)
    model.train()
    model.to(device)
    
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dataset = Dataset(datadir)
    
    return model, criterion, optimizer, dataset


if __name__ == '__main__':
    run_save_path = '/work3/s214629/runs/RCNN'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>>>> Using device: ", DEVICE)
    model, criterion, optimizer, dataset = simple_setup(device = DEVICE, datadir = '/work3/s214629/preprocessed_data/RCNN_preproc/segments')
    
    train_model(model, criterion, optimizer, dataset, num_epochs=50, device=DEVICE, batch_size=150, run_save_path=run_save_path)