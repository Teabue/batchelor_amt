import torch
import pandas as pd
from midiutil.MidiFile import MIDIFile
import mido
import json
import os
from utils.dataloader import Dataset
from utils.model import Resnext50
import tqdm
import librosa
import numpy as np
import cv2
import mido


    
def process_audio_to_segments(audio_path: os.PathLike, sr: int = 22050, hop_length: int = 512, transform: str = "cqt", tempo: int = 60000):
    """Basically the same as preprocessing but with less information saved

    Args:
        audio_path (os.PathLike): _description_
        sr (int, optional): _description_. Defaults to 22050.
        hop_length (int, optional): _description_. Defaults to 512.
        transform (str, optional): _description_. Defaults to "cqt".
        tempo (int, optional): _description_. Defaults to 60000.

    Returns:
        _type_: _description_
    """
    aud, _ = librosa.load(audio_path, sr=sr)
    
    # -------------------- Get amplitude after transformation -------------------- #
    if transform == "cqt":
        aud = librosa.cqt(aud, sr=sr, hop_length=hop_length) 
    elif transform == "stft":
        aud = librosa.stft(aud, sr=sr, hop_length=hop_length)
    else:
        ValueError("Invalid choice of preprocess method.")
    
    S_db = librosa.amplitude_to_db(np.abs(aud), ref=np.max)
    
    # ----------------------------- Get onset frames ----------------------------- #

    o_env = librosa.onset.onset_strength(S=S_db, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    
    
    # ----------------------------- Get segment indexes ----------------------------- #
    # Get times from S_db
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    
    # Calculate the seconds per sixteenth note
    interval = tempo / (4 * 1000000) # 4 because we want to have sixteenth notes #TODO: make this a config perhaps?
    
    # Calculate the number of indexes that fit into each time interval
    indexes_per_interval = int(np.round(interval / np.diff(times).mean()))
    
    # Get segment indexes
    segments = [(i,i+indexes_per_interval) for i in range(0, len(S_db[0]), indexes_per_interval)]
    
    # --------------------------- Set up output audio folder -------------------------- #
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    audio_dir = os.path.join(os.path.dirname(__file__), "processed_audio", audio_name)
    
    img_dir = os.path.join(audio_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    
    # ------------- Save segment information: csv and spectrogram img ------------ #
    
    # Create dataframe
    df_processed_data = pd.DataFrame(columns=['segment_img_path', 'onset'])
    
    print("_________________________ Processing audio _________________________")
    for segment_start_idx, segment_end_idx in tqdm.tqdm(segments, total=len(segments)):

        # Check if there were any onsets on the segment
        onset = np.any((segment_start_idx <= onset_frames) & (onset_frames < segment_end_idx))
        
        # ---------------------------- Save segment image ---------------------------- #
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
        
        img_path = os.path.join(img_dir, f'{audio_name}_start-{segment_start_idx}_end-{segment_end_idx}.png')
        cv2.imwrite(img_path, img)
        
        # ----------------------------- Update dataframe ----------------------------- #
        # Add the segment to the dataframe
        df_processed_data.loc[len(df_processed_data)] = {'segment_img_path': img_path, 'onset': onset}
    
    # --------------------------------- Save csv --------------------------------- #
    
    return df_processed_data
    
    


def midi_inference_new_song(model_path: os.PathLike, audio_path: os.PathLike, nr_pitches = 128, device = 'cpu', bpm = 120, sr = 22050, hop_length = 512, transform = "cqt"):
    model = Resnext50(nr_pitches)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    dataset = Dataset("", nr_pitches=nr_pitches)

    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # ------------- Save segment information: csv and spectrogram img ------------ #
    song_segments = process_audio_to_segments(audio_path, sr=sr, hop_length=hop_length, transform=transform, tempo=mido.bpm2tempo(bpm))

    # ------------------------------ Midifile stuff ------------------------------ #
    mf = MIDIFile(1)     # only 1 track
    
    track = 0   # the only track

    time = 0    # start at the beginning
    
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)

    # add some notes
    channel = 0
    volume = 100
    duration = 1/4 #sixteenth note, hopefully :))
    
    previous_pitches = []
    set_in_pitch = {} 
    # ------------------------------------ end ----------------------------------- #
    
    for _, segment in tqdm.tqdm(song_segments.iterrows(), total=len(song_segments)):
        # ------------------------------ Get model preds ----------------------------- #
        img, _ = dataset.get_inference_segment("[]", segment['segment_img_path'])
        img = img.to(device)
        
        outputs = model(img.unsqueeze(0))
        preds = (outputs.squeeze(0) > 0.7).to(device)
        
        pred_pitches = torch.nonzero(preds).squeeze(1)
        
        onset = segment['onset']
        
        # ----------------------------- Add to pred midi ----------------------------- #        
        for pitch in pred_pitches:
            pitch = int(pitch.item()) # convert to float
            
            if pitch in previous_pitches and not onset:
                set_in_pitch[pitch][1] += duration
                continue
            elif pitch in previous_pitches and onset:
                set_in_pitch[pitch] = [time, duration]
                mf.addNote(track, channel, pitch, time, duration, volume)
            elif pitch not in previous_pitches:
                set_in_pitch[pitch] = [time, duration]
                mf.addNote(track, channel, pitch, set_in_pitch[pitch][0], set_in_pitch[pitch][1], volume)
            else:
                AssertionError("You shall not pass! - Because you ended up here which you shouldn't.")
        
        previous_pitches = pred_pitches
        
        time += duration
        
    # write it to disk
    with open(audio_name + "_pred.mid", 'wb') as outf:
        mf.writeFile(outf)
                    
    
    
    
    

def midi_inference_test_set(model_path: os.PathLike, csv_segment_dir: os.PathLike, nr_pitches = 128, device = 'cpu'):
    """Inference on test set

    Args:
        model_path (os.PathLike): _description_
        csv_segment_dir (os.PathLike): _description_
        nr_pitches (int, optional): _description_. Defaults to 128.
        device (str, optional): _description_. Defaults to 'cpu'.
    """

    model = Resnext50(nr_pitches)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    dataset = Dataset(csv_segment_dir, nr_pitches=nr_pitches)


    df = pd.read_csv(os.path.join(csv_segment_dir,"test.csv"))

    songs = df['audio_path'].unique()


    for song_path in songs:
        song_segments = df.loc[df['audio_path'] == song_path]
        correct = torch.tensor(0).to(device)
        
        
        # ------------------------------ Midifile stuff ------------------------------ #
        mf = MIDIFile(1)     # only 1 track
        mfGT = MIDIFile(1)     # GT
        
        track = 0   # the only track

        time = 0    # start at the beginning
        
        mf.addTrackName(track, time, "Sample Track")
        mf.addTempo(track, time, int(mido.tempo2bpm(song_segments.iloc[0]['audio_tempo'])))


        mfGT.addTrackName(track, time, "Sample Track")
        mfGT.addTempo(track, time, int(mido.tempo2bpm(song_segments.iloc[0]['audio_tempo'])))
        
        # add some notes
        channel = 0
        volume = 100
        duration = 1/4 #sixteenth note, hopefully :))

        previous_pitches_GT = []
        set_in_pitch_GT = {} 
        
        previous_pitches = []
        set_in_pitch = {} 
        # ------------------------------------ end ----------------------------------- #
        
        for _, segment in tqdm.tqdm(song_segments.iterrows(), total=len(song_segments)):
            # ------------------------------ Get model preds ----------------------------- #
            img, labels = dataset.get_inference_segment(segment['segment_midi_pitches'], segment['segment_img_path'])
            img = img.to(device)
            labels = labels.to(device)
            
            outputs = model(img.unsqueeze(0))
            preds = (outputs.squeeze(0) > 0.95).to(device)
            correct += torch.sum(preds == labels)
            
            pred_pitches = torch.nonzero(preds).squeeze(1)
            
            
            # ----------------------- The same for both GT and pred ---------------------- #
            onset = segment['onset']
            
            # ------------------------------ Add to GT midi ------------------------------ #
            pitches = json.loads(segment['segment_midi_pitches'])

            for pitch in pitches:
                if pitch in previous_pitches_GT and not onset:
                    set_in_pitch_GT[pitch][1] += duration
                    continue
                elif pitch in previous_pitches_GT and onset:
                    set_in_pitch_GT[pitch] = [time, duration]
                    mfGT.addNote(track, channel, pitch, time, duration, volume)
                elif pitch not in previous_pitches_GT:
                    set_in_pitch_GT[pitch] = [time, duration]
                    mfGT.addNote(track, channel, pitch, set_in_pitch_GT[pitch][0], set_in_pitch_GT[pitch][1], volume)
                else:
                    AssertionError("You shall not pass! - Because you ended up here which you shouldn't.")
            
            previous_pitches_GT = pitches
            
            # ----------------------------- Add to pred midi ----------------------------- #        
            for pitch in pred_pitches:
                pitch = int(pitch.item()) # convert to float
                
                if pitch in previous_pitches and not onset:
                    set_in_pitch[pitch][1] += duration
                    continue
                elif pitch in previous_pitches and onset:
                    set_in_pitch[pitch] = [time, duration]
                    mf.addNote(track, channel, pitch, time, duration, volume)
                elif pitch not in previous_pitches:
                    set_in_pitch[pitch] = [time, duration]
                    mf.addNote(track, channel, pitch, set_in_pitch[pitch][0], set_in_pitch[pitch][1], volume)
                else:
                    AssertionError("You shall not pass! - Because you ended up here which you shouldn't.")
            
            previous_pitches = pred_pitches
            
            
            # ------------------------------- Same for both ------------------------------ #
            time += duration
            
            
            
        # write it to disk
        with open(os.path.splitext(os.path.basename(song_segments.iloc[0]['audio_path']))[0] + "_GT.mid", 'wb') as outf:
            mfGT.writeFile(outf)
            
            
        # write it to disk
        with open(os.path.splitext(os.path.basename(song_segments.iloc[0]['audio_path']))[0] + "_pred.mid", 'wb') as outf:
            mf.writeFile(outf)
            
        print(f"Accuracy: {correct.item() / len(song_segments)*nr_pitches}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "../classification/models/model_best.pth"
    # model_path = r"C:\University\6th_semester\Bachelor_proj\classification\models\model_best.pth"
    
    inference_on = "new_song" # "new_song" or "test_set"
    
    if inference_on == "new_song":
        AUDIO_PATH = "../river.mp3"
        # AUDIO_PATH = r"C:\University\6th_semester\Bachelor_proj\river.mp3"
        BPM = 65
        midi_inference_new_song(model_path=model_path, audio_path=AUDIO_PATH, nr_pitches=128, device=device, bpm=BPM)
        
    elif inference_on == "test_set":
        CSV_SEGMENT_DIR = "../data_process/output/segments"
        # CSV_SEGMENT_DIR = r"C:\University\6th_semester\Bachelor_proj\data_process\output\segments"
        
        midi_inference_test_set(model_path=model_path, csv_segment_dir=CSV_SEGMENT_DIR, nr_pitches=128, device=device)
    
    
    
    