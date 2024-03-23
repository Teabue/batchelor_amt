import torch 
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import copy
import json

# Preload all the spectrograms outside of class so it doesn't get copied for each dataset split


class TransformerDataset(torch.utils.data.Dataset):
    data_df: pd.DataFrame
    song_dict: dict[str, np.ndarray] = {} # we keep it on CPU in order to save GPU memory
    
    def __init__(self, data_dir: os.PathLike) -> None:
        self.data_dir = data_dir
        
        if not TransformerDataset.song_dict:
            self.load_spectrograms()
        
    def load_spectrograms(self):
        song_names = []
        for split in ['train', 'val', 'test']:
            data_df = pd.read_csv(os.path.join(self.data_dir, split, 'labels.csv'), index_col=0)
            song_names.extend(data_df['song_name'].unique())
        # NOTE: We don't have to further unique it as train/val/test were split up by songs, not their segments
        
        # Preload the spectrograms because it's faster than loading them on the fly
        for song_name in song_names:
            spectrogram = np.load(os.path.join(self.data_dir, 'spectrograms', f'{song_name}.npy'))
            TransformerDataset.song_dict[song_name] = spectrogram
            
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        df_idx = self.data_df.loc[idx]
        song_name = df_idx.song_name
        labels = df_idx.labels

        spectrogram = TransformerDataset.song_dict[song_name]
        spectrogram = spectrogram[:, int(df_idx.sequence_start_idx): int(df_idx.sequence_end_idx)]
        spectrogram = spectrogram.T # Transpose, so that when indexing [idx] we get a sequence of frequency amplitudes for time step idx
        spectrogram = torch.from_numpy(spectrogram)
        
        labels = json.loads(labels)
        labels = torch.tensor(labels)
        
        return spectrogram, labels
    
    def pad_collate_fn(self, batch):
        """
        Collate function for the DataLoader. Pads the spectrograms and labels to the same length
        """
        spectrograms, labels = zip(*batch)
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=-1)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        return spectrograms, labels
    
    def get_split(self, split: str, **kwargs) -> DataLoader:
        dataset_split = copy.copy(self)
        dataset_split.data_df = pd.read_csv(os.path.join(self.data_dir, split, 'labels.csv'))
        
        return DataLoader(dataset_split, collate_fn=self.pad_collate_fn, **kwargs)
    

        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = TransformerDataset(r'/work3/s214629/preprocessed_data')
    
    batch_size = 150
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # plt.rcParams['figure.figsize'] = [18, 6]
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        for batch in train_loader:
            spectrogram, labels = batch[0], batch[1]
            labels.to(device)
            spectrogram.to(device)
            for batch_idx in range(batch_size):
                plot_spec = spectrogram[batch_idx]
                l = labels[batch_idx]
                
                
                
                # plot_spec = spectrogram[0].numpy()
                # l = labels[0].numpy()
                
                # plt.imshow(plot_spec, cmap='viridis')
                # plt.show()
            
        
