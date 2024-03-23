import numpy as np
import torch
import os
import pandas as pd
import yaml
from tqdm import tqdm
import json

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from torch.nn.utils.rnn import pad_sequence

from utils.vocabularies import Vocabulary
from utils.preprocess_song import Song
from utils.model import Transformer


def create_midi_from_model_events(events, bpm_tempo, output_dir='', onset_only=False):
    """Don't use onset_only, it's truly shit
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    tempo = bpm2tempo(bpm_tempo)
    track.append(MetaMessage('key_signature', key='A'))
    track.append(MetaMessage('set_tempo', tempo=tempo))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))

    delta_time = 0
    offset_onset = None
    for event in events:
        if event[0] == 'PAD' or event[0] == 'SOS' or event[0] == 'ET':
            continue
        elif event[0] == 'EOS':
            offset_onset = None
        elif event[0] == 'time_shift':
            delta_time += second2tick(event[1]/100, mid.ticks_per_beat, tempo)
        elif event[0] == 'offset_onset':
            offset_onset = event[1]
        elif event[0] == 'pitch':
            if offset_onset is None:
                # These pitches are from before the ET token
                continue
            if offset_onset == 0:
                # Just a glorified note-off event. I'm doing this because maestro did it first >:(
                if not onset_only:
                    track.append(Message('note_off', channel=1, note=event[1], velocity=64, time=delta_time))
            elif offset_onset == 1:
                if onset_only:
                    track.append(Message('note_on', channel=1, note=event[1], velocity=100, time=delta_time))
                    track.append(Message('note_off', channel=1, note=event[1], velocity=100, time=delta_time+128))
                else:
                    track.append(Message('note_on', channel=1, note=event[1], velocity=100, time=delta_time))
            else:
                raise ValueError('offset_onset should be 0 or 1')
            
            delta_time = 0

    track.append(MetaMessage('end_of_track'))

    mid.save(os.path.join(output_dir,'river_pred.midi'))

if __name__ == '__main__':
    
    new_song_path = '/zhome/5d/a/168095/batchelor_amt/test_songs/river.mp3'
    test_new_song = True
    bpm_tempo = 65
    
    # ----------------------------- Choose test song ----------------------------- #
    song_name = 'MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_01_Track01_wav'
    data_dir = '/work3/s214629/preprocessed_data_best'
    test_preprocessing_works = False
    # ------------------------------- Choose model ------------------------------- #
    
    run_path = '/work3/s214629/run_a100_hope3/'
    model_name = 'model_best.pth'
    
    # --------------------------------- Run stuff -------------------------------- #
    with open("Transformer/configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # This is scuffed, but ignore it lolz
    with open("Transformer/configs/vocab_config.yaml", 'r') as f:
        vocab_configs = yaml.safe_load(f)
    vocab = Vocabulary(vocab_configs)
    vocab.define_vocabulary()
    tgt_vocab_size = vocab.vocab_size



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not test_new_song:
        spectrogram = np.load(os.path.join(data_dir, 'spectrograms', f'{song_name}.npy'))
        df = pd.read_csv(os.path.join(data_dir, 'test', 'labels.csv'))
        df = df[df['song_name'] == song_name] # get all segments of the song

        sequences = []
        for idx, row in df.iterrows():
            sequence = spectrogram[:, int(row['sequence_start_idx']): int(row['sequence_end_idx'])]
            sequence = sequence.T
            sequence = torch.from_numpy(sequence)
            sequences.append(sequence)
        
        spectrograms = pad_sequence(sequences, batch_first=True, padding_value=-1)
        
    else: 
        # TODO WARNING THIS IS NOT FLEXIBLE
        with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
            preprocess_config = yaml.safe_load(f)
        song = Song(new_song_path,preprocess_config)
        spectrograms = song.preprocess_inference_new_song()

    spectrograms = spectrograms.to(device)
    if not test_preprocessing_works:
        model = Transformer(config['n_mel_bins'], tgt_vocab_size, config['d_model'], config['num_heads'], config['num_layers'], config['d_ff'], config['max_seq_length'], config['dropout'], device)
        model.load_state_dict(torch.load(os.path.join(run_path, 'models', model_name)))
        model.to(device)
        model.eval()

        # Prepare the sequences
        all_sequence_events = []
        # This might be dumb, Idk yet 
        for sequence_spec in tqdm(spectrograms, total=len(spectrograms), desc='Generating predictions'):
            # torch.tensor([[1]]) because SOS token is 1 
            output = model.get_sequence_predictions(sequence_spec.unsqueeze(0), torch.tensor([[1]], device=device), config['max_seq_length'])
            output = output.squeeze()
            events = vocab.translate_sequence_token_to_events(output.tolist())
            all_sequence_events.extend(events)
    elif not test_new_song:
        # Create a midi based off of the ground truths
        all_sequence_events = []
        for idx, row in df.iterrows():
            labels = json.loads(row['labels'])
            events = vocab.translate_sequence_token_to_events(labels)
            all_sequence_events.extend(events)
    else:
        raise ValueError('test_new_song and test_preprocessing_works cannot both be True.')
        
    if not test_new_song:
        for root, dirs, files in os.walk('/work3/s214629/maestro-v3.0.0/maestro-v3.0.0'):
            if song_name + '.midi' in files:
                bpm_tempo = MidiFile(os.path.join(root, song_name + '.midi')).tracks[0][0].tempo
                break

    create_midi_from_model_events(all_sequence_events, bpm_tempo, onset_only=False)



