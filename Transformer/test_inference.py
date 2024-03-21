from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
import numpy as np
import torch
from utils.model import Transformer
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import yaml
from utils.vocabularies import Vocabulary
import json


def create_midi_from_model_events(events, bpm_tempo, output_dir=''):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    tempo = tempo=bpm2tempo(bpm_tempo)
    track.append(MetaMessage('key_signature', key='Dm'))
    track.append(MetaMessage('set_tempo', tempo=tempo))
    track.append(MetaMessage('time_signature', numerator=6, denominator=8))

    delta_time = 0
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
                track.append(Message('note_on', channel=1, note=event[1], velocity=0, time=delta_time))
            elif offset_onset == 1:
                track.append(Message('note_on', channel=1, note=event[1], velocity=100, time=delta_time))
            else:
                raise ValueError('offset_onset should be 0 or 1')
            
            delta_time = 0

    track.append(MetaMessage('end_of_track'))

    mid.save(os.path.join(output_dir,'predict_model.midi'))

if __name__ == '__main__':

    song_name = 'ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--1'
    data_dir = '/work3/s214629/preprocessed_data'
    run_path = '/work3/s214629/run_a100'
    model_name = 'model_best_epoch-3.pth'
    test_preprocessing_works = False

    with open("Transformer/configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # This is scuffed, but ignore it lolz
    with open("Transformer/configs/vocab_config.yaml", 'r') as f:
        vocab_configs = yaml.safe_load(f)
    vocab = Vocabulary(vocab_configs)
    vocab.define_vocabulary()
    tgt_vocab_size = vocab.vocab_size -1



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    spectrogram = np.load(os.path.join(data_dir, 'spectrograms', f'{song_name}.npy'))
    df = pd.read_csv(os.path.join(data_dir, 'test', 'labels.csv'))
    df = df[df['song_name'] == song_name] # get all segments of the song

    if not test_preprocessing_works:
        model = Transformer(config['n_mel_bins'], tgt_vocab_size, config['d_model'], config['num_heads'], config['num_layers'], config['d_ff'], config['max_seq_length'], config['dropout'], device)
        model.load_state_dict(torch.load(os.path.join(run_path, 'models', model_name)))
        model.to(device)
        model.eval()

        # Prepare the sequences
        sequences = []
        for idx, row in df.iterrows():
            sequence = spectrogram[:, row['sequence_start_idx']: row['sequence_end_idx']]
            sequence = sequence.T
            sequence = torch.from_numpy(sequence)
            sequences.append(sequence)
            
        spectrograms = pad_sequence(sequences, batch_first=True, padding_value=-1)
        spectrograms = spectrograms.to(device)

        # torch.ones since SOS is token 1
        SOS_start_tokens = torch.ones((spectrograms.size(0), 1), dtype=torch.int32, device=device)

        outputs = model(spectrograms, SOS_start_tokens)

        all_sequence_events = []
        for output in outputs:
            events = vocab.translate_sequence_token_to_events(output.tolist())
            all_sequence_events.extend(events)
    else:
        # Create a midi based off of the ground truths
        all_sequence_events = []
        for idx, row in df.iterrows():
            labels = json.loads(row['labels'])
            events = vocab.translate_sequence_token_to_events(labels)
            all_sequence_events.extend(events)
        
    
    for root, dirs, files in os.walk('/work3/s214629/maestro-v3.0.0/maestro-v3.0.0'):
        if song_name + '.midi' in files:
            bpm_tempo = MidiFile(os.path.join(root, song_name + '.midi')).tracks[0][0].tempo
            break
    print(all_sequence_events)
    create_midi_from_model_events(all_sequence_events, bpm_tempo)



