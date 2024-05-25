import numpy as np
import torch
import os
import pandas as pd
import yaml
from tqdm import tqdm
import json

from fractions import Fraction
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
    
def translate_events_to_sheet_music(event_sequence: list[tuple[str, int]],
                                    bpm: int,
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
    # TODO: Use the downbeat events to estimate the time signature
    # treble_staff.timeSignature = meter.TimeSignature('4/4')
    # bass_staff.timeSignature = meter.TimeSignature('4/4')
    
    df = _prepare_data_frame(event_sequence)
    print("Dataframe prepared!")
    
    # Convert time to quarternote length
    df['duration'] = (df['offset'] - df['onset'])
    last_recorded_downbeat = None
    beats_per_bar = 0
    cur_bpm = 0
    for idx, row in df.iterrows():
        
        if row['full_note'] == "Downbeat":
            # Update the number of beats in a bar
            if last_recorded_downbeat is not None:
                
                # If a new time signature is detected
                if beats_per_bar != (row['onset'] - last_recorded_downbeat):
                    beats_per_bar = row['onset'] - last_recorded_downbeat
                    
                    multiplier = 1
                    while (beats_per_bar * multiplier) % 1 != 0:
                        multiplier += 1
                    
                    nominator = int(beats_per_bar * multiplier)
                    denominator = 2**(multiplier + 1)
                
                    treble_staff.insert(last_recorded_downbeat, meter.TimeSignature(f'{nominator}/{denominator}'))
                    bass_staff.insert(last_recorded_downbeat, meter.TimeSignature(f'{nominator}/{denominator}'))

            last_recorded_downbeat = row['onset']
            
            if row['bpm'] != cur_bpm:
                # Add the metronome mark to the score
                metronome_mark = tempo.MetronomeMark(number = row['bpm'])
                treble_staff.insert(last_recorded_downbeat, metronome_mark)
                bass_staff.insert(last_recorded_downbeat, metronome_mark)
                
                cur_bpm = row['bpm']
            
            continue
                
        xml_note = note.Note(row['full_note'])
        xml_note.duration = duration.Duration(row['duration'])
        xml_note.offset = row['onset']
        
        if row['bpm'] != cur_bpm:
            # Add the metronome mark to the score
            metronome_mark = tempo.MetronomeMark(number = row['bpm'])
            treble_staff.insert(xml_note.offset, metronome_mark)
            bass_staff.insert(xml_note.offset, metronome_mark)
            
            cur_bpm = row['bpm']
        
        if row['duration'] == 0:
            xml_note = xml_note.getGrace()
        
        # Evenly distribute between treble and bass cleff - could be optimized
        # Depending on the predicted duration of the df, the insert method should act accordingly
        staff = treble_staff if xml_note.octave >= 4 else bass_staff
        ns = list(staff.getElementsByOffset(row['onset'], classList=[note.Note, chord.Chord]))
        ds = [float(n.duration.quarterLength) for n in ns]
        if ns and xml_note.duration.quarterLength in ds:
            notes_to_chordify = np.where(np.isclose(ds, xml_note.duration.quarterLength))[0]
            [staff.remove(ns[n]) for n in notes_to_chordify]
            chord_to_insert = chord.Chord(list(ns[n] for n in notes_to_chordify) + [xml_note])
            
            staff.insert(xml_note.offset, chord_to_insert)
        else:
            staff.insert(xml_note.offset, xml_note)
    
    print("Done with adding to the streams. Now we make the score!") 
    # Connect the streams to a score
    score = stream.Score()        

    # Add the staffs to the score
    score.insert(0, treble_staff)
    score.insert(0, bass_staff)
    piano = layout.StaffGroup([treble_staff, bass_staff], symbol='brace')
    score.insert(0, piano)
    
    print("We have a score! We fill in rests")
    # Fill in rests where there is empty space
    for part in score.parts:
        part.makeRests(fillGaps=True, inPlace=True)

    print("We propose a key")
    # Change pitches to the analyzed key 
    # NOTE: If the song changes key throughout, it may screw-up the key analysis
    proposed_key = score.analyze('key')
    
    print("We update the pitches if the key has flats")
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

    print("We can show the score and write it to a file")
    # score.show()
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
    df = pd.DataFrame(columns=['full_note', 'onset', 'offset', 'bpm'])
    
    beat = 0
    eos_beats = 0
    last_downbeat_onset = 0
    bpm = 0
    notes_to_concat: dict[str, list[int]] = {}
    et_switch = False
    onset_switch = False
    for idx, (event, value) in enumerate(event_sequence):
        if isinstance(value, str):
            value = int(value)
        
        if event == "SOS":
            if eos_beats != beat:
                eos_beats = beat
            et_switch = True
        
        if event == "tempo":
            bpm = value
        
        elif event == 'downbeat':
            last_downbeat_onset = beat
            df = pd.concat([df, pd.DataFrame([{'full_note': "Downbeat", 'onset': last_downbeat_onset, 'offset': last_downbeat_onset, 'bpm': bpm}])], ignore_index=True)
        
        elif event == "ET":
            et_switch = False
        
        elif event == "EOS":# Keeps track of absolute beat number
            eos_beats = beat
        
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
                if note_ in notes_to_concat.keys():
                    print(f"Note ({note_}) is set to onset again before offset!")
                    notes_to_concat[note_].append(beat)
                else:
                    # Save the note with the corresponding onset time
                    notes_to_concat[note_] = [beat]
            
            elif not et_switch and not onset_switch:
                if note_ not in notes_to_concat.keys():
                    # This shouldn't happen but I'll allow it and skip it
                    print(f"Note ({note_}) not found in list.")
                    continue
                
                if len(notes_to_concat[note_]) > 1:
                    print(f"Note ({note_}) has more than one onset! " +
                          f"Taking the last one and offsetting it to {beat}")
                    
                    df_onset = notes_to_concat[note_].pop(-1)
                else:
                    # Remove from the dict and add to the dataframe
                    df_onset = notes_to_concat.pop(note_)[0]
                
                df = pd.concat([df, pd.DataFrame([{'full_note': note_, 'onset': df_onset, 'offset': beat, 'bpm': bpm}])], ignore_index=True)
        
        elif event == "beat":
            # Find indices of the tuplets
            sub_beats = np.arange(0, 1, vocab_configs['subdivision'])
            tup_beats = np.arange(0, 1, Fraction(vocab_configs['tuplet_subdivision']))
            common = np.intersect1d(sub_beats, tup_beats)
            tup_beats = np.setdiff1d(tup_beats, common)
            tuplet_indices = (np.searchsorted(sub_beats, tup_beats) + np.arange(0, len(tup_beats))).tolist() # We add the range to account for the fact that we are adding tuplets
            
            notes_per_beat = len(sub_beats) + len(tup_beats)
            
            # Convert the time shift to quarternote fractions
            if value % notes_per_beat in tuplet_indices:
                new_beat = value // notes_per_beat + tup_beats[tuplet_indices.index(value % notes_per_beat)]
            
            # Not a tuplet
            else:
                # How many tuplets on the way to the current bar + how many on the current bar
                decrement = value // notes_per_beat * len(tup_beats) + np.sum((value % notes_per_beat) > np.asarray(tuplet_indices)) 
                new_beat = (value - decrement) * vocab_configs['subdivision']
            
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
    inference_dir = "inference_songs"
    new_song_name = "pirate_ensemble_105.mp3"
    new_song_path = os.path.join(inference_dir, new_song_name) # '/zhome/5d/a/168095/batchelor_amt/test_songs/river.mp3'
    test_new_song = False
    
    # ----------------------------- Choose test song ----------------------------- #
    song_name = "Pokemon_Ruby_Sapphire_Emerald_Ending_Credits_Theme" # 'MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_01_Track01_wav'
    data_dir = "preprocessed_data/21-05-24" # '/work3/s214629/preprocessed_data_best'
    test_preprocessing_works = True
    # ------------------------------- Choose model ------------------------------- #
    
    run_path = "" # '/work3/s214629/run_a100_hope3/'
    model_name = '07-05-24_musescore.pth'
    
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
    only_save = False
    
    if not saved:
        if not test_new_song:
            spectrogram = np.load(os.path.join(data_dir, 'spectrograms', f'{song_name}.npy'))
            
            for i in ['train', 'val', 'test']:              
                df = pd.read_csv(os.path.join(data_dir, i, 'worker_0.csv'))
                
                if (df['song_name'] == song_name).any():
                    break
                
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
            
            np.save(os.path.join(inference_dir, new_song_name), all_sequence_events)
        
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
        all_sequence_events = np.load(os.path.join(inference_dir, new_song_name + '.npy'), allow_pickle=True)
    
    token_sequence = [1, 
                      133, 108-12*4, 135, 
                      132, 108-12*4, 137, 
                      133, 22+12*2, 26+12*2, 170, 
                      132, 22+12*2, 
                      133, 37+12, 32+12, 175, 
                      132, 26+12*2, 37+12, 32+12, 13+12*3, 
                      2, 0]
    # events = vocab.translate_sequence_token_to_events(token_sequence)
    
    if not only_save:
        bpm_tempo = 100 # 65
        translate_events_to_sheet_music(all_sequence_events, bpm = bpm_tempo, output_dir=f"{song_name}")
    # create_midi_from_model_events(all_sequence_events, bpm_tempo, output_dir="/Users/helenakeitum/Desktop", onset_only=False)