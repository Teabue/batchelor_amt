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

def test_preprocessing(data_dir: str):
    songs_dir = os.path.join(data_dir, "spectrograms")
    songs = [os.path.splitext(file)[0] for file in os.listdir(songs_dir)]
    
    for song_name in songs:
        spectrogram = np.load(os.path.join(data_dir, 'spectrograms', f'{song_name}.npy'))
        
        for i in ['train', 'val', 'test']:              
            df = pd.read_csv(os.path.join(data_dir, i, 'labels.csv'))
            
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
        spectrograms = spectrograms.to(device)
        
        # Create a sheet based off of the ground truths
        sequence_events = []
        for idx, row in df.iterrows():
            labels = json.loads(row['labels'])
            events = vocab.translate_sequence_token_to_events(labels)
            sequence_events.extend(events)
        
        translate_events_to_sheet_music(sequence_events, output_dir=f"{song_name}")
    
def inference(init_bpm: int, inference_dir: str, new_song_name: str, saved_seq: bool, just_save_seq: bool = False):
    new_song_path = os.path.join(inference_dir, new_song_name) # '/zhome/5d/a/168095/batchelor_amt/test_songs/river.mp3'
    if not saved_seq:
        
        # TODO WARNING THIS IS NOT FLEXIBLE
        with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
            preprocess_config = yaml.safe_load(f)
        song = Song(new_song_path, preprocess_config)
        spectrogram = song.compute_spectrogram()
        
        model = Transformer(config['n_mel_bins'], tgt_vocab_size, config['d_model'], config['num_heads'], config['num_layers'], config['d_ff'], config['max_seq_length'], config['dropout'], device)
        # model.load_state_dict(torch.load(os.path.join(run_path, 'models', model_name), map_location=device))
        model.to(device)
        model.eval()

        # Prepare the sequences
        sequence_events = []
        
        # ----------------------------- Slice spectrogram dynamically ----------------------------- #
        cur_bpm = init_bpm
        cur_frame = 0
        while cur_frame < spectrogram.shape[1]:
            spec_slice, new_frame = song.preprocess_inference_new_song(spectrogram, cur_frame = cur_frame, bpm = cur_bpm)
            spec_slice = spec_slice.T
            spec_slice = spec_slice.to(device)
            
            cur_frame = new_frame
            
            # torch.tensor([[1]]) because SOS token is 1 
            output = model.get_sequence_predictions(spec_slice.unsqueeze(0), torch.tensor([[1, cur_bpm]], device=device), config['max_seq_length'])
            output = output.squeeze()
            events = vocab.translate_sequence_token_to_events(output.tolist())
            sequence_events.extend(events)
            
            if np.any(events > vocab.vocabulary['tempo'][1]):
                cur_bpm = events[np.where(events > vocab.vocabulary['tempo'][1])[0][-1]]
                
        np.save(os.path.join(inference_dir, new_song_name), sequence_events)
    
    else:
        sequence_events = np.load(os.path.join(inference_dir, new_song_name))
        
    if not just_save_seq:
        translate_events_to_sheet_music(sequence_events, output_dir=f"{new_song_name}")
        
        
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
    
    df = _prepare_data_frame(event_sequence)
    df = df.sort_values(by=['onset'])
    print("Dataframe prepared!")
    
    # Calculate how long each measuer should be
    measure_durations = np.diff(df[df['event'].isin(['SOS', 'Downbeat', 'EOS'])]['onset'])
    
    # Convert time to quarternote length
    df['duration'] = (df['offset'] - df['onset'])
    beats_per_bar = 0
    i = 0
    beats_per_bar = 0
    for idx, row in df.iterrows():
        
        if row['event'] in ['SOS', 'Downbeat']:
            # TODO: Consider making Measures objects to correctly display pickups (measurenumber = 0)
            # If a new time signature is detected
            if beats_per_bar != measure_durations[i]:
                beats_per_bar = measure_durations[i]
                
                multiplier = 1
                while (beats_per_bar * multiplier) % 1 != 0:
                    multiplier += 1
                
                nominator = int(beats_per_bar * multiplier)
                denominator = 2**(multiplier + 1)
            
                treble_staff.insert(row['onset'], meter.TimeSignature(f'{nominator}/{denominator}'))
                bass_staff.insert(row['onset'], meter.TimeSignature(f'{nominator}/{denominator}'))
            
            i += 1
    
        elif row['event'] == "Tempo":
            metronome_mark = tempo.MetronomeMark(number = row['offset'])
            metronome_mark.placement = 'above'
            treble_staff.insert(row['onset'], metronome_mark)
            
        elif row['event'] == "EOS":
            final_barline = bar.Barline('final')
            treble_staff.append(final_barline)
            bass_staff.append(final_barline)
        
        else:  
            if len(row['event']) > 1:
                xml_note = chord.Chord(row['event'])
            else:      
                xml_note = note.Note(row['event'][0])
            
            xml_note.duration = duration.Duration(row['duration'])
            xml_note.offset = row['onset']
            
            if row['duration'] == 0:
                xml_note = xml_note.getGrace()
            
            # Evenly distribute between treble and bass cleff - could be optimized
            if xml_note.isChord:
                highest_note = xml_note.sortAscending().pitches[-1]
                lowest_note = xml_note.sortAscending().pitches[0]
                
                # Decide which staff to insert the chord into based on the pitches of the highest and lowest notes
                if highest_note.octave < 4:
                    staff = bass_staff
                elif lowest_note.octave >= 4:
                    staff = treble_staff
                else:
                    staff = treble_staff # TODO: Split the chord into notes again
            else:       
                staff = treble_staff if xml_note.octave >= 4 else bass_staff
            
            # Insert into notes if the onset is already occupied
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
    df = pd.DataFrame(columns=['event', 'onset', 'offset'])
    df = pd.concat([df, pd.DataFrame([{'event': "SOS", 'onset': 0, 'offset': 0}])], ignore_index=True)
    
    beat = 0
    eos_beats = 0
    last_downbeat_onset = 0
    cur_bpm = 0
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
            if value != cur_bpm:
                df = pd.concat([df, pd.DataFrame([{'event': "Tempo", 'onset': beat, 'offset': value}])], ignore_index=True) # NOTE offset is the bpm
                cur_bpm = value
        
        elif event == 'downbeat':
            last_downbeat_onset = beat
            df = pd.concat([df, pd.DataFrame([{'event': "Downbeat", 'onset': last_downbeat_onset, 'offset': last_downbeat_onset}])], ignore_index=True)
        
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
                
                # If note(s) with the same onset and offset exists in the dataframe, they are a chord - only notes are lists
                pos_chord = df[(df['onset'] == df_onset) & (df['offset'] == beat) & (df['event'].apply(lambda x: isinstance(x, list)))]
                if not pos_chord.empty:
                    df.loc[pos_chord.index, 'event'] = pos_chord['event'].apply(lambda x: x + [note_]) # Add the note to the chord
                else:
                    df = pd.concat([df, pd.DataFrame([{'event': [note_], 'onset': df_onset, 'offset': beat}])], ignore_index=True)
        
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
    
    # Add last downbeat to indicate end-of-score
    df = pd.concat([df, pd.DataFrame([{'event': "EOS", 'onset': beat, 'offset': beat}])], ignore_index=True)
            
    return df


def _get_note_value(pitch):
    # NOTE perhaps less octaves. 
    # a piano has 8 octaves and musescore doesn't even allow that many
    notes = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    octaves = np.arange(11)
    
    return (notes[pitch % 12], str(octaves[pitch // 12] - 1))

if __name__ == '__main__':
    # ----------------------------- For inference ----------------------------- #
    inference_dir = "inference_songs"
    new_song_name = "pirate_ensemble_105.mp3"
    
    # ----------------------------- For preprocessing ----------------------------- #
    data_dir = "preprocessed_data"
    
    # ------------------------------- Choose model ------------------------------- #
    run_path = "" # '/work3/s214629/run_a100_hope3/'
    model_name = '07-05-24_musescore.pth' 
    
    # --------------------------------- Run stuff -------------------------------- #
    with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
        pre_configs = yaml.safe_load(f)
    
    with open("Transformer/configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # This is scuffed, but ignore it lolz
    with open("Transformer/configs/vocab_config.yaml", 'r') as f:
        vocab_configs = yaml.safe_load(f)
    vocab = Vocabulary(vocab_configs)
    vocab.define_vocabulary(pre_configs['max_beats'])
    tgt_vocab_size = vocab.vocab_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    new_song = True
    init_bpm = 130
    if new_song:
        inference(init_bpm = init_bpm, inference_dir = inference_dir, new_song_name = new_song_name, saved_seq = False, just_save_seq = False)
    else:
        test_preprocessing(data_dir)