import torch
import tqdm
import os
import numpy as np
import torch.nn as nn

from utils.dataloader import Dataset
from utils.model import Resnext50

def train_model(model, criterion, optimizer, dataset: Dataset, num_epochs=100, device='cuda', batch_size=20, nr_pitches=128):
    

    for epoch in range(num_epochs):
        
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(train_loader, total = len(train_loader), \
                         desc=f'Train: Loss: [{1}], Epochs: {epoch}/{num_epochs}', leave = False)    

        losses = []
        for images, labels in pbar:#iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
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
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            if i > 200:
                break
                
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            pbar.set_description(f'Test: Loss: [{loss.item():.4f}], Epochs: {epoch}/{num_epochs}')
            losses.append(loss.item())
            
        # Save model if loss is better
        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'model_best.pth'))  


def simple_setup(lr = 0.001, wd = 0.0001, device = 'cuda', nr_pitches=128, datadir = 'data_process/output/segments'):
    
    model = Resnext50(nr_pitches)
    model.train()
    model.to(device)
    
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dataset = Dataset(datadir)
    
    return model, criterion, optimizer, dataset


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>>>> Using device: ", DEVICE)
    model, criterion, optimizer, dataset = simple_setup(device = DEVICE, datadir = 'data_process/output/segments')
    
    train_model(model, criterion, optimizer, dataset, num_epochs=50, device=DEVICE, batch_size=150)