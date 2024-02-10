import librosa
import mido
import numpy as np
import os
import yaml
from glob import glob
import random
import pandas as pd
from typing import List
import tqdm
import cv2


class Data_Preprocessor():
    def __init__(self, CONFIG_PATH: os.PathLike) -> None:
        with open(CONFIG_PATH, 'r') as f:
            CONFIG = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = CONFIG
        
        # Save segments as images?
        self.save_segment_img = CONFIG['save_spectrogram_imgs']
        
        # Audio sampling configs (we can add n_bins, but we'll see)
        self.sr = CONFIG['sample_rate']
        self.hop_length = CONFIG['hop_length']
        self.transform = CONFIG['transform']
        
        # Splits
        self.train_split = CONFIG['train_split']
        self.val_split = CONFIG['val_split']
        self.test_split = CONFIG['test_split']
        self.random_seed = CONFIG['random_seed']
        
        # Output folders
        self.output_dir = CONFIG['output_dir']
        self.output_img_dir = CONFIG['output_img_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_img_dir, exist_ok=True)
        
        # Set these in child class
        self.files: List[str] = None
        self.audio_format: str = None
    
    def get_splits(self):
        # Shuffle files; we can do this better through hashing, but meh
        random.seed(self.random_seed)
        random.shuffle(self.files)
        
        # Split files
        train_files = self.files[:int(len(self.files) * self.train_split)]
        val_files = self.files[int(len(self.files) * self.train_split):int(len(self.files) * (self.train_split + self.val_split))]
        test_files = self.files[int(len(self.files) * (self.train_split + self.val_split)):]
        
        return train_files, val_files, test_files
    
    
    # ---------------------------------------------------------------------------- #
    #                 Call per audio file - in order to save memory                #
    # ---------------------------------------------------------------------------- #
    
    def audio_get_transformed(self, audio_name: str):
        # NOTE: Assumes that librosa.load can load the audio file format!!!
        aud_path = os.path.join(audio_name + self.audio_format)
        
        aud, _ = librosa.load(aud_path, sr=self.sr)
        
        if self.transform == "cqt":
            aud = librosa.cqt(aud, sr=self.sr, hop_length=self.hop_length) 
        elif self.transform == "stft":
            aud = librosa.stft(aud, sr=self.sr, hop_length=self.hop_length)
        else:
            ValueError("Invalid choice of preprocess method.")
        
        return librosa.amplitude_to_db(np.abs(aud), ref=np.max)
    
    def audio_get_onsets(self, aud: np.ndarray):
        """Librosa implementation of onset detection.

        Args:
            aud (np.ndarray): S_db from librosa.amplitude_to_db of audio - i.e. from output of audio_get_transformed
        """
        o_env = librosa.onset.onset_strength(S=aud, sr=self.sr)
        # times = librosa.times_like(o_env, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.sr)
        
        return onset_frames
    
    def audio_get_ground_truth(self, audio_name: str, S_db: np.ndarray):
        """Should list of lists with ground truth midi pitches
        where the parent list has the same length of time indexes of the transformed audio
        """
        # This will vary from dataset to dataset, so do this in child class
        NotImplementedError("Implement this in child class")
    
    
    def audio_get_tempo(self, audio_name: str):
        """Should return tempo like in a midi file see: https://mido.readthedocs.io/en/stable/files/midi.html#tempo-and-time-resolution
        basicalle tempo is microseconds per quarternote
        """
        # WARNING: Be careful whether the time signature has 4 as the denominator when calculating tempo, if not, OOPS :D 
        NotImplementedError("Implement this in child class")
    
    def audio_concat_segments(self, tempo: int, 
                              S_db: np.ndarray, 
                              onset_frames: np.ndarray, 
                              ground_truths: List[List[int]], 
                              df_preprocessed_data: pd.DataFrame, 
                              audio_name: str,
                              split: str) -> pd.DataFrame:

        # Get times from S_db
        times = librosa.times_like(S_db, sr=self.sr, hop_length=self.hop_length)
        
        # Calculate the seconds per sixteenth note
        interval = tempo / (4 * 1000000) # 4 because we want to have sixteenth notes #TODO: make this a config perhaps?
        
        # Calculate the number of indexes that fit into each time interval
        indexes_per_interval = int(np.round(interval / np.diff(times).mean()))
        
        # Get segment indexes
        segments = [(i,i+indexes_per_interval) for i in range(0, len(S_db[0]), indexes_per_interval)]
        
        for segment_start_idx, segment_end_idx in segments:
            # Concatenate ground truths to include all midi pitches in the segment
            segment_midi_pitches = list(np.unique([item for sublist in ground_truths[segment_start_idx:segment_end_idx] for item in sublist])) #Note: this may be slow, I'm too lazy to look at it now
            
            # Check if there were any onsets on the segment
            onset = np.any((segment_start_idx <= onset_frames) & (onset_frames < segment_end_idx))
            
            
            audio_path = os.path.join(audio_name + self.audio_format)
            
            if self.save_segment_img:
                img_path = self.segment_save_img(S_db, segment_start_idx, segment_end_idx, audio_name, split)
            else:
                img_path = None
                
            # Add the segment to the dataframe
            df_preprocessed_data.loc[len(df_preprocessed_data)] = {'audio_path': audio_path,
                                                                'audio_sr': self.sr, 
                                                                'audio_hop_length': self.hop_length, 
                                                                'audio_transform': self.transform,
                                                                'audio_tempo': tempo,
                                                                'segment_start_idx': segment_start_idx, 
                                                                'segment_end_idx(not_including)': segment_end_idx, 
                                                                'segment_midi_pitches': segment_midi_pitches, 
                                                                'segment_img_path': img_path,
                                                                'onset': onset}

        return df_preprocessed_data


    
    # ---------------------------------------------------------------------------- #
    #                          Segment specific functions                          #
    # ---------------------------------------------------------------------------- #
    def segment_save_img(self, S_db: np.ndarray, segment_start_idx: int, segment_end_idx: int, audio_name: str, split: str) -> os.PathLike:        
        # Get segment
        seg_aud = S_db[:, segment_start_idx:segment_end_idx]
        
        def min_max_scale(seg_aud, aud, min_val=0, max_val=255):
            array_std = (seg_aud - aud.min()) / (aud.max() - aud.min())
            array_scaled = array_std * (max_val - min_val) + min_val
            return array_scaled

        # min-max scale to fit inside 8-bit range
        img = min_max_scale(seg_aud, S_db, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image

        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        
        # Save image
        audio_name = os.path.splitext(os.path.basename(audio_name))[0]
        img_dir = os.path.join(self.output_img_dir, split)
        os.makedirs(img_dir, exist_ok=True)
        
        img_path = os.path.join(img_dir, f'{audio_name}_start-{segment_start_idx}_end-{segment_end_idx}.png')
        cv2.imwrite(img_path, img)
        return img_path
    
    # ---------------------------------------------------------------------------- #
    #                     Call this function to preprocess data                    #
    # ---------------------------------------------------------------------------- #
    def save_preprocessed_data(self):
        split_names = ["train", "val", "test"]
        splits = self.get_splits()
        
        for split_name, split_files in zip(split_names, splits):
            df_preprocessed_data = pd.DataFrame(columns=['audio_path','audio_sr', 'audio_hop_length', 'audio_transform', 'audio_tempo', 'segment_start_idx', 'segment_end_idx(not_including)', 'segment_midi_pitches', 'segment_img_path', 'onset'])
            
            print(f"Preprocessing {split_name} split")
            for audio_name in tqdm.tqdm(split_files, total=len(split_files)):
                S_db = self.audio_get_transformed(audio_name)
                onset_frames = self.audio_get_onsets(S_db)
                ground_truths = self.audio_get_ground_truth(audio_name, S_db)
                tempo = self.audio_get_tempo(audio_name)
                
                df_preprocessed_data =  self.audio_concat_segments(tempo, S_db, onset_frames, ground_truths, df_preprocessed_data, audio_name, split_name)
                
            df_preprocessed_data.to_csv(os.path.join(self.output_dir, f'{split_name}.csv'), index=False)
    


# ---------------------------------------------------------------------------- #
#                               MAPS preprocessor                              #
# ---------------------------------------------------------------------------- #

class MAPS_Preprocessor(Data_Preprocessor):
    def __init__(self, CONFIG_PATH: os.PathLike):
        super().__init__(CONFIG_PATH)
        
        # MAPS files
        self.audio_format = ".wav"
        
        data_folder = self.config['MAPS_datapath']
        self.files = [y.replace('.wav', '') for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.wav'))]
        
        
    def audio_get_ground_truth(self,  audio_name: str, S_db: np.ndarray):
        txt_path = os.path.join(audio_name + ".txt")
        
        txt_data = pd.read_csv(txt_path, delim_whitespace=True)
        txt_data = txt_data.to_dict("list")
        
        times = librosa.times_like(S_db, sr=self.sr, hop_length=self.hop_length)
        
        ground_truths = []
        for t in times:
            current_pitches = [p for p, o, off in zip(txt_data["MidiPitch"], txt_data["OnsetTime"], txt_data["OffsetTime"]) if o <= t < off]
            ground_truths.append(current_pitches)
            
        return ground_truths
    
    def audio_get_tempo(self, audio_name: str):
        mid_path = os.path.join(audio_name + ".mid")
        midi_data = mido.MidiFile(mid_path, clip=True)
        tempo = midi_data.tracks[0][0].tempo # WARNING: Me assume tempo denominator is 4 brrrrrrrrrrr :))
        
        return tempo
        
if __name__ == '__main__':
    path = "config.yaml"
    # path = r'C:\University\6th_semester\Bachelor_proj\data_process\config.yaml'
    
    preprocessor = MAPS_Preprocessor(path)
    preprocessor.save_preprocessed_data()

        
        
        

        
