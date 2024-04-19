import numpy as np
import torch
import os
import pandas as pd
import yaml
from tqdm import tqdm
import json

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from music21 import *
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
            delta_time += int(round(second2tick(int(event[1])/100, mid.ticks_per_beat, tempo)))
        elif event[0] == 'offset_onset':
            offset_onset = int(event[1])
        elif event[0] == 'pitch':
            if offset_onset is None:
                # These pitches are from before the ET token
                continue
            if offset_onset == 0:
                # Just a glorified note-off event. I'm doing this because maestro did it first >:(
                if not onset_only:
                    track.append(Message('note_off', channel=1, note=int(event[1]), velocity=64, time=delta_time))
            elif offset_onset == 1:
                if onset_only:
                    track.append(Message('note_on', channel=1, note=int(event[1]), velocity=100, time=delta_time))
                    track.append(Message('note_off', channel=1, note=int(event[1]), velocity=100, time=delta_time+128))
                else:
                    track.append(Message('note_on', channel=1, note=int(event[1]), velocity=100, time=delta_time))
            else:
                raise ValueError('offset_onset should be 0 or 1')
            
            delta_time = 0

    track.append(MetaMessage('end_of_track'))

    mid.save(os.path.join(output_dir,'river_pred.midi'))
    
def translate_events_to_sheet_music(event_sequence: list[tuple[str, int]], bpm: int,
                                    output_dir = "/Users/helenakeitum/Desktop/output.xml"):
    """example: [('timeshift', 200), ('onset', 0), ('pitch', 60), ('offset', 100), ('pitch', 62), ('onset', 100)]

    Args:
        events (list[tuple[str, int]]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    treble_staff = stream.PartStaff()
    bass_staff = stream.PartStaff()
    
    treble_staff.clef = clef.TrebleClef()
    bass_staff.clef = clef.BassClef()
    
    # TODO: Hardcoded. Make it flexible
    treble_staff.timeSignature = meter.TimeSignature('4/4')
    bass_staff.timeSignature = meter.TimeSignature('4/4')
    
    # Add the metronome mark to the score
    metronome_mark = tempo.MetronomeMark(number=bpm)
    treble_staff.insert(0, metronome_mark)
    bass_staff.insert(0, metronome_mark)
    
    df = _prepare_data_frame(event_sequence)
    print("Dataframe prepared!")
    
    # Convert time to quarternote length
    df['duration'] = (df['offset'] - df['onset'])
    
    for idx, row in df.iterrows():
        if row['duration'] == 0:
            continue
        if row['full_note'] == 'Rest':
            xml_note = note.Rest()
            xml_note.duration = duration.Duration(row['duration'])
        else:
            xml_note = note.Note(row['full_note'])
            xml_note.duration = duration.Duration(row['duration'])
        
        # Evenly distribute between treble and bass cleff - could be optimized
        # Depending on the predicted duration of the df, the insert method should act accordingly
        if (row['onset'] in df.drop(idx)['onset'].values and row['duration'] in df[df['onset'] == row['onset']].drop(idx)['duration'].values):
            if row['octave'] == '-':
                treble_staff.insertIntoNoteOrChord(row['onset'], xml_note)
                bass_staff.insertIntoNoteOrChord(row['onset'], xml_note)
            elif row['octave'] >= '4':
                treble_staff.insertIntoNoteOrChord(row['onset'], xml_note)
            else:
                bass_staff.insertIntoNoteOrChord(row['onset'], xml_note)
        else:
            if row['octave'] == '-':
                # Let MuseScore handle the rests
                continue
                treble_staff.insert(row['onset'], xml_note)
                bass_staff.insert(row['onset'], xml_note)
            elif row['octave'] >= '4':
                treble_staff.insert(row['onset'], xml_note)
            else:
                bass_staff.insert(row['onset'], xml_note)
        
    # Connect the streams to a score
    score = stream.Score()        

    # Add the staffs to the score
    score.insert(0, treble_staff)
    score.insert(0, bass_staff)
    piano = layout.StaffGroup([treble_staff, bass_staff], symbol='brace')
    score.insert(0, piano)

    # Change pitches to the analyzed key 
    # NOTE: If the song changes key throughout, it may screw-up the key analysis
    proposed_key = score.analyze('key')
    
    # Only update the pitches if the key signature includes flats
    if proposed_key.sharps < 0:
        score = _update_pitches(score, proposed_key)

    # Add the key signature
    key_signature = key.KeySignature(proposed_key.sharps)
    treble_staff.keySignature = key_signature
    bass_staff.keySignature = key_signature

    # NOTE: These MAY BE useful in the future
    # score.makeNotation(inPlace=True)
    # treble_staff.makeAccidentals(inPlace = True)
    # bass_staff.makeAccidentals(inPlace = True)

    score.show()
    score.write('musicxml', fp=output_dir)


def _update_pitches(score, key_sig) -> stream.Score:
    for n in score.recurse().notes:
        if isinstance(n, chord.Chord):
            for n_ in n:
                if n_.pitch.accidental is not None and key_sig.accidentalByStep(n_.transpose(1).step) is not None:
                    n_.pitch = n_.pitch.getEnharmonic()
        else:
            if n.pitch.accidental is not None and key_sig.accidentalByStep(n.transpose(1).step) is not None:
                n.pitch = n.pitch.getEnharmonic()

    return score

def _prepare_data_frame(event_sequence: list[tuple[str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(columns=['full_note', 'pitch', 'octave', 'onset', 'offset'])
    
    beat = 0
    eos_beats = 0
    notes_to_concat = {}
    et_switch = False
    onset_switch = False
    for event, value in event_sequence:
        if isinstance(value, str):
            value = int(value)
        
        if event == "SOS":
            et_switch = True
        
        if event == "ET":
            et_switch = False
        
        elif event == "EOS":
            # Keeps track of absolute beat number
            eos_beats += 4 # TODO: It needs to extract the no of beats of a bar
        
        elif event == "offset_onset" and value == 1:
            onset_switch = True
            et_switch = False
            
        elif event == 'offset_onset' and value == 0:
            onset_switch = False
            et_switch = False
        
        elif event == "pitch":
            note_value, octave_value = _get_note_value(value)
            note_ = note_value + octave_value
            
            # If the et_switch is one, we don't want to save the note or offset it prematurily
            if not et_switch and onset_switch:
                # Save the note with the corresponding onset time
                notes_to_concat[note_] = beat
            
            elif not et_switch and not onset_switch:
                if note_ not in notes_to_concat.keys():
                    # This shouldn't happen but I'll allow it and skip it
                    print(f"Note ({note_}) not found in list")
                    continue
                
                # Remove from the dict and add to the dataframe
                df_onset = notes_to_concat.pop(note_)
                df = pd.concat([df, pd.DataFrame([{'full_note': note_, 'pitch': note_value, 'octave': octave_value, 'onset': df_onset, 'offset': beat}])], ignore_index=True)
        
        elif event == "beat":
            notes_per_beat = (1 / vocab_configs['subdivision'] + 2) # +2 because there are always 2 eight note tuplets
            
            # Find indices of the eight tuplets
            sub_beats = np.arange(0, 1, vocab_configs['subdivision'])
            first_tuplet = np.argwhere(sub_beats > 1/3)[0][0]
            second_tuplet = np.argwhere(sub_beats > 2/3)[0][0] + 1 # + 1 because we have already found the first tuplet
            tuplet_indices = np.array([first_tuplet, second_tuplet])
            
            # Convert the time shift to quarternote fractions
            if value % notes_per_beat == first_tuplet:
                new_beat = value // notes_per_beat + 1/3
            elif value % notes_per_beat == second_tuplet:
                new_beat = value // notes_per_beat + 2/3
            
            # Not a tuplet
            else:
                # How many tuplets on the way to the current bar + how many on the current bar
                decrement = value // notes_per_beat * 2 + np.sum((value % notes_per_beat) > tuplet_indices) # np.floor(value % notes_per_beat / 4)
                new_beat = (value - decrement) * vocab_configs['subdivision']
            
            # If we are not onsetting and skipping in time, we have a rest!
            if not onset_switch:
                df = pd.concat([df, pd.DataFrame([{'full_note': "Rest", 'pitch': "-", 'octave': "-", 'onset': beat, 'offset': new_beat + eos_beats}])], ignore_index=True)

            beat = new_beat + eos_beats
            
    return df


def _get_note_value(pitch):
    # NOTE perhaps less octaves. 
    # a piano has 8 octaves and musescore doesn't even allow that many
    notes = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    octaves = np.arange(11)
    
    return (notes[pitch % 12], str(octaves[pitch // 12] - 1))

if __name__ == '__main__':
    
    new_song_path = "/Users/helenakeitum/Desktop/midis/pirate_ensemble_105.mp3" # '/zhome/5d/a/168095/batchelor_amt/test_songs/river.mp3'
    test_new_song = False
    bpm_tempo = 100 # 65
    
    # ----------------------------- Choose test song ----------------------------- #
    song_name = "I_Hate_to_Admit_It___-_Bang_Chan_" # 'MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_01_Track01_wav'
    data_dir = "preprocessed_data_best" # '/work3/s214629/preprocessed_data_best'
    test_preprocessing_works = True
    # ------------------------------- Choose model ------------------------------- #
    
    run_path = "" # '/work3/s214629/run_a100_hope3/'
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
    saved = False
    
    if not saved:
        if not test_new_song:
            spectrogram = np.load(os.path.join(data_dir, 'spectrograms', f'{song_name}.npy'))
            df = pd.read_csv(os.path.join(data_dir, 'val', 'labels.csv'))
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
            model.load_state_dict(torch.load(os.path.join(run_path, 'models', model_name), map_location=device))
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
            
        # if not test_new_song:
        #     for root, dirs, files in os.walk('/work3/s214629/maestro-v3.0.0/maestro-v3.0.0'):
        #         if song_name + '.midi' in files:
        #             bpm_tempo = MidiFile(os.path.join(root, song_name + '.midi')).tracks[0][0].tempo
        #             break
    else:
        all_sequence_events = np.load('/Users/helenakeitum/Desktop/saved_seq_events.npy', allow_pickle=True)
    
    token_sequence = [1, 
                      133, 108-12*4, 135, 
                      132, 108-12*4, 137, 
                      133, 22+12*2, 26+12*2, 170, 
                      132, 22+12*2, 
                      133, 37+12, 32+12, 175, 
                      132, 26+12*2, 37+12, 32+12, 13+12*3, 
                      2, 0]
    # events = vocab.translate_sequence_token_to_events(token_sequence)
    
    translate_events_to_sheet_music(all_sequence_events, bpm = bpm_tempo)
    # create_midi_from_model_events(all_sequence_events, bpm_tempo, output_dir="/Users/helenakeitum/Desktop", onset_only=False)