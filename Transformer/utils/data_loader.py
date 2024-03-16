import torch 
import pandas as pd
import os
import librosa 
import numpy as np
import cv2
from torch.utils.data import DataLoader
import copy
import json
import torchvision.transforms as transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    data_df: pd.DataFrame
    
    def __init__(self, data_dir: os.PathLike) -> None:
        self.data_dir = data_dir
        
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        df_idx = self.data_df.loc[idx]
        song_name = df_idx.song_name
        labels = df_idx.labels

        spectrogram = np.load(os.path.join(self.data_dir, 'spectrograms', f'{song_name}.npy'))
        spectrogram = spectrogram[:, df_idx.sequence_start_idx: df_idx.sequence_end_idx]
        spectrogram = torch.from_numpy(spectrogram)
        
        labels = json.loads(labels)
        labels = torch.tensor(labels)
        
        return spectrogram, labels
    
    
    def get_split(self, split: str, **kwargs) -> DataLoader:
        dataset_split = copy.copy(self)
        dataset_split.data_df = pd.read_csv(os.path.join(self.data_dir, split, 'labels.csv'))
        return DataLoader(dataset_split, **kwargs)
    

        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = Dataset(r'C:\University\batchelor_amt\Transformer\preprocessed_data')
    
    batch_size = 6
    train_loader = dataset.get_split('test', batch_size=batch_size, shuffle=False)
    
    plt.rcParams['figure.figsize'] = [18, 6]

    for batch in train_loader:
        spectrogram, labels = batch[:, 0], batch[:, 1]
        # labels.to(device)
        # spectrogram.to(device)
        for batch_idx in range(batch_size):
            spectrogram = spectrogram[0].numpy()
            labels = labels[0].numpy()
            
            plt.imshow(spectrogram, cmap='viridis')
            plt.show()
        
        
