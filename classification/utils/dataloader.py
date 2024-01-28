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
    
    def __init__(self, data_dir: os.PathLike, img_size = 84, transform=None, nr_pitches=128):

        self.nr_pitches = nr_pitches
        self.data_dir = data_dir
        self.img_size = img_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        
        audio_path,audio_sr,audio_hop_length,audio_transform,audio_tempo,segment_start_idx,segment_end_idx,segment_midi_pitches,onset = self.data_df.loc[idx]
        
        aud, sr = librosa.load(audio_path, sr=audio_sr)
        
        if audio_transform == 'cqt':
            aud = librosa.cqt(aud, sr=audio_sr, hop_length=audio_hop_length)
        elif audio_transform == 'stft':
            aud = librosa.stft(aud, sr=audio_sr, hop_length=audio_hop_length)
        else:
            ValueError("Invalid choice of preprocess method.")
            
        aud = librosa.amplitude_to_db(np.abs(aud), ref=np.max)
        
        # Get segment
        seg_aud = aud[:, segment_start_idx:segment_end_idx]
        
        def min_max_scale(seg_aud, aud, min_val=0, max_val=255):
            array_std = (seg_aud - aud.min()) / (aud.max() - aud.min())
            array_scaled = array_std * (max_val - min_val) + min_val
            return array_scaled

        # min-max scale to fit inside 8-bit range
        img = min_max_scale(seg_aud, aud, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image

        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #idk, I like rgb :))
        img = Image.fromarray(img)
        img = self.transform(img)
        
        labels = torch.zeros(self.nr_pitches)
        labels[json.loads(segment_midi_pitches)] = 1 

        return img, labels
    
    
    def get_split(self, split: str, **kwargs) -> DataLoader:
        dataset_split = copy.copy(self)
        dataset_split.data_df = pd.read_csv(os.path.join(self.data_dir, f'{split}.csv'))
        
        return DataLoader(dataset_split, **kwargs)
        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = Dataset('data_process/output/segments')
    
    train_loader = dataset.get_split('test', batch_size=6, shuffle=False)
    
    plt.rcParams['figure.figsize'] = [18, 6]

    for images, segment_midi_pitches in iter(train_loader):
        for img in images:
            plt.imshow(img)
            plt.show()
