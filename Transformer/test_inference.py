import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import yaml
import xml.etree.ElementTree as ET

from fractions import Fraction
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from music21 import converter, stream, note, chord, duration, meter, tempo, spanner, clef, layout, key, bar, metadata

from utils.vocabularies import Vocabulary
from utils.preprocess_song import Song
from utils.model import Transformer

class Inference:
    def __init__(self, vocab: Vocabulary, configs: dict[str, str | None], 
                 dirs: dict[str, str | None], song_name: str, model_name: str = None):
        
        self.vocab = vocab
        
        self.pre_config = configs['preprocess']
        self.vocab_config = configs['vocab']
        self.train_config = configs['train']
        
        self.data_dir = dirs['data'] # For preprocessed data
        self.audio_dir = dirs['audio']
        self.output_dir = dirs['out']
        
        self.song_name = song_name
        self.model_name = model_name
    
    def plot_pianoroll(self, m21_stream: stream.Score, alpha: float = 0.5) -> None:
        # Flatten the stream to handle all notes and chords
        flat_stream = m21_stream.flatten()
        
        # Extract all notes and chords with their offsets
        elements = [(elem.offset, elem) for elem in flat_stream.getElementsByClass((note.Note, chord.Chord))]
        
        def inference_stats(notes):
            # Separate the pitches into two sets based on their color
            pred, gt = set(), set()
            for tup in notes:
                p = tup[1]
                elements = p.notes if p.isChord else [p]
                for n in elements:
                    if n.style.color == "red":
                        pred.add((n.pitch.midi, p.offset, p.offset + p.quarterLength))
                        pred_highest_time = p.offset + p.quarterLength
                    elif n.style.color == "blue":
                        gt.add((n.pitch.midi, p.offset, p.offset + p.quarterLength))
                        gt_highest_time = p.offset + p.quarterLength
                    else:
                        pred.add((n.pitch.midi, p.offset, p.offset + p.quarterLength))
                        gt.add((n.pitch.midi, p.offset, p.offset + p.quarterLength))  
                        
                        if p.offset + p.quarterLength > gt_highest_time:
                            gt_highest_time = p.offset + p.quarterLength
                        if p.offset + p.quarterLength > pred_highest_time:
                            pred_highest_time = p.offset + p.quarterLength

            # Compute the unions
            pred_duration = sum(end - start for _, start, end in pred)
            gt_duration = sum(end - start for _, start, end in gt)
            
            # Compute the total true negative area
            gt_total_area = gt_highest_time * vocab.vocabulary['pitch'][0].min_max_values[1]
            pred_total_area = pred_highest_time * vocab.vocabulary['pitch'][0].min_max_values[1]
            total_area = max(gt_total_area, pred_total_area)
            
            # Compute the intersection
            overlap_duration = sum(min(end, gt_end) - max(start, gt_start)
                            for midi, start, end in pred
                            for gt_midi, gt_start, gt_end in gt
                            if midi == gt_midi and start < gt_end and end > gt_start)
            
            union = pred_duration + gt_duration - overlap_duration
            
            false_positive = (pred_duration - overlap_duration) / total_area
            false_negative = (gt_duration - overlap_duration) / total_area
            true_positive = overlap_duration / total_area
            true_negative = (total_area - union) / total_area
            
            # Compute the relevant statistics
            iou = overlap_duration / union
            sensitivity = true_positive / (true_positive + false_negative)
            false_negative_rate = false_negative / (true_positive + false_negative)
            false_positive_rate = false_positive / (false_positive + true_negative)
            specificity = true_negative / (false_positive + true_negative)
            
            return iou, sensitivity, false_negative_rate, false_positive_rate, specificity

        # Compute the iou
        iou, sens, fnr, fpr, spec = inference_stats(elements)
        
        plt.ioff()
        _, ax = plt.subplots()

        # Find y-axis range as where notes are actually playing
        all_pitches = [p.midi for elem in flat_stream.notesAndRests for p in elem.pitches if isinstance(elem, (note.Note, chord.Chord))]
        if all_pitches:
            min_pitch = min(all_pitches)
            max_pitch = max(all_pitches)
        else:
            min_pitch, max_pitch = 21, 108  # Default range if no notes

        # Add the notes
        for offset, elem in elements:
            ns = elem.notes if elem.isChord else [elem]
            for n in ns:
                color = n.style.color if n.style.color else 'black'
                ax.broken_barh([(offset, elem.quarterLength)], (n.pitch.midi - 0.5, 1), facecolors=color, alpha=alpha)
        
        ax.set_xlabel('Time (quarter lengths)')
        ax.set_ylabel('Pitch')
        
        # Set the y-axis range to the relevant pitches
        ax.set_ylim(min_pitch - 1, max_pitch + 1)
        y_ticks = range(min_pitch, max_pitch + 1)
        ax.set_yticks(y_ticks)

        # Set y-axis labels to note names
        y_tick_labels = [note.Note(midi).nameWithOctave for midi in y_ticks]
        ax.set_yticklabels(y_tick_labels)

        # Add alternating background colors for rows
        for i in range(min_pitch, max_pitch, 2):
            ax.axhspan(i - 0.5, i + 0.5, facecolor='white', alpha=0.3, linewidth=0)
            ax.axhspan(i + 0.5, i + 1.5, facecolor='lightgrey', alpha=0.3, linewidth=0)

        # Add dotted horizontal lines for clarity
        for tick in y_ticks:
            ax.axhline(tick - 0.5, color='grey', linestyle='dotted', linewidth=0.5)
        
        # Combine statistics into one string
        stats_text = f"IoU: {iou:.2f}\nSensitivity: {sens:.2f}\nFalse Negative Rate: {fnr:.2f}\nFalse Positive Rate: {fpr:.2f}\nSpecificity: {spec:.2f}"

        # Define properties of the text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Adjust these values to move the position of the text box
        x_position = 1.05
        y_position = 0.5

        # Add text box
        ax.text(x_position, y_position, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Save the image
        save_dir = os.path.join(self.output_dir, "piano_roll")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, self.song_name), bbox_inches='tight')
        plt.close()

    def test_preprocessing(self, train_val_test: list[str] = ['train', 'val', 'test']) -> None:
        
        for i in train_val_test:              
            df = pd.read_csv(os.path.join(self.data_dir, i, 'labels.csv'))
            songs = pd.unique(df['song_name'])
            
            for song_name in songs:
            
                df_song = df[df['song_name'] == song_name] # get all segments of the song
                
                if df_song.empty:
                    print(f"The song {song_name} could not be found")
                
                # Create a sheet based off of the ground truths
                sequence_events = []
                for _, row in df_song.iterrows():
                    labels = json.loads(row['labels'])
                    events = self.vocab.translate_sequence_token_to_events(labels)
                    sequence_events.extend(events)
                
                output_dir = os.path.join(self.output_dir, song_name)
                self.translate_events_to_sheet_music(sequence_events, output_dir)
            
    def inference(self, init_bpm: int, saved_seq: bool = False, just_save_seq: bool = False) -> stream.Score:
        
        tgt_vocab_size = self.vocab.vocab_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not saved_seq:
            song = Song(self.audio_dir, self.pre_config)
            if self.data_dir is None:
                spectrogram = song.compute_spectrogram()
            else:
                spectrogram = np.load(os.path.join(self.data_dir, "spectrograms", f"{self.song_name}.npy"))
            
            model = Transformer(self.train_config['n_mel_bins'], tgt_vocab_size, 
                                self.train_config['d_model'], self.train_config['num_heads'], 
                                self.train_config['num_layers'], self.train_config['d_ff'], 
                                self.train_config['max_seq_length'], self.train_config['dropout'], device)
            
            model.load_state_dict(torch.load(os.path.join('models', self.model_name), map_location=device))
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
                token_bpm = self.vocab.vocabulary['tempo'][0].translate_value_to_token(cur_bpm)
                
                # torch.tensor([[1]]) because SOS token is 1 
                output = model.get_sequence_predictions(spec_slice.unsqueeze(0), torch.tensor([[1, token_bpm]], device=device), self.pre_config['max_sequence_length'])
                output = output.squeeze()
                events = self.vocab.translate_sequence_token_to_events(output.tolist())
                sequence_events.extend(events)
                
                if np.any([event[0] == "tempo" for event in events[2:]]):
                    cur_bpm = events[np.where([event[0] == "tempo" for event in events])[0][-1]][1]
            
            os.makedirs(os.path.join(self.output_dir, 'seq_predictions'), exist_ok=True)        
            
            if sequence_events:
                np.save(os.path.join(self.output_dir, 'seq_predictions', self.song_name), sequence_events)
        
        else:
            sequence_events = np.load(os.path.join(self.output_dir, 'seq_predictions', f'{self.song_name}.npy'))
            
        if not just_save_seq:
            output_dir = os.path.join(self.output_dir, 'sheet_predictions')
            os.makedirs(output_dir, exist_ok=True)   
            return self.translate_events_to_sheet_music(sequence_events, os.path.join(output_dir, self.song_name))

    def overlap_gt_and_pred(self, gt_score: stream.Score, pred_score: stream.Score) -> None:
        os.makedirs(os.path.join(self.output_dir, "overlap"), exist_ok=True)
        
        # Boolean capital letters for (pitch, onset, offset) - TTT is just default black
        cm_colors = {"GT": "blue", "PRED": "red"}
        
        # I wish I knew this function existed before...
        gt_score = stream.tools.removeDuplicates(gt_score)
        # pred_score = stream.tools.removeDuplicates(pred_score)
        
        # Convert scores to a time series with only the available predicted elements
        gt_tree = gt_score.asTimespans(classList=(note.Note, chord.Chord, meter.TimeSignature, tempo.MetronomeMark))
        pred_tree = pred_score.asTimespans(classList=(note.Note, chord.Chord, meter.TimeSignature, tempo.MetronomeMark))
        gt_time = 0
        
        # Prepare new empty streams
        t_staff = stream.PartStaff(id = "Treble")
        b_staff = stream.PartStaff(id = "Bass")
        
        def _compare_elements(e1, e2):
            if isinstance(e1, note.NotRest):
                return [[[m1.midi == m2.midi, e1.quarterLength == e2.quarterLength] for m1 in e1.pitches] for m2 in e2.pitches]
            elif isinstance(e1, meter.TimeSignature):
                return e1.ratioString == e2.ratioString
            elif isinstance(e1, tempo.MetronomeMark):
                return e1.number == e2.number
            return False
        
        for time_point in pred_tree.allOffsets():
            
            # Search for false negatives
            while time_point > gt_time:
                gt_elements = gt_tree.elementsStartingAt(gt_time)
                
                # Insert the elements into the overlap stream
                for gt_e in gt_elements:
                    gt_e.element.offset = gt_time
                    gt_e.element.style.color = cm_colors["GT"]
                    
                    if isinstance(gt_e.element, tempo.MetronomeMark):
                        t_staff.insert(gt_time, gt_e.element)
                    elif isinstance(gt_e.element, meter.TimeSignature):
                        t_staff.insert(gt_time, gt_e.element)
                        b_staff.insert(gt_time, gt_e.element)
                    elif isinstance(gt_e.element, note.NotRest):
                        self._assign_and_insert_into_staff(gt_e.element, t_staff, b_staff)
                    
                # Update the ground truth time to see if we missed anything
                gt_time = gt_tree.getPositionAfter(gt_time)
                if gt_time is None:
                    gt_time = time_point + 1
            
            # Go through each element in the prediction stream
            processed_time_sig_offsets = set()
            not_rest_mask = {} # Mask for events in the gt stream
            for e in pred_tree.elementsStartingAt(time_point):
                pred_mask = {}
                event = e.element
                event.offset = time_point
                
                # If the event is a time signature and its offset has already been processed, skip it
                if isinstance(event, meter.TimeSignature):
                    if event.offset in processed_time_sig_offsets:
                        continue
                    processed_time_sig_offsets.add(time_point)
                
                # We need to prepare the mask for the event in case there are no gt's
                if isinstance(event, note.NotRest):
                    pred_mask[event] = np.full(len(event.pitches), False)
                else:
                    pred_mask[event] = np.array([False])
                    
                # Go through the ground truth elements at the current onset
                for gt_e in gt_tree.elementsStartingAt(time_point):
                    gt_event = gt_e.element
                    gt_event.offset = time_point
                    
                    if (gt_event.classes[0] != event.classes[0]):
                        # If one is a chord and the other is a note for example, it's fine
                        if not (isinstance(gt_event, note.NotRest) and isinstance(event, note.NotRest)):
                            continue
                    
                    # Compare the elements
                    equal = _compare_elements(event, gt_event)
                    
                    # Update the matches of the event and gt event
                    if isinstance(gt_event, note.NotRest):
                        matched_gt_notes = np.any(np.all(equal, axis = 2), axis = 1)
                        not_rest_mask[gt_event] = np.full(len(gt_event.pitches), False)
                        not_rest_mask[gt_event][matched_gt_notes] = True

                        matched_event_notes = np.any(np.all(equal, axis = 2), axis = 0)
                        pred_mask[event][matched_event_notes] = True

                    else:
                        if equal:
                            pred_mask[event] = np.array([True])
                
                # Insert predictions into the stream
                for e, mask in pred_mask.items():
                    
                    if isinstance(e, note.NotRest):
                        for idx, n in enumerate(e if e.isChord else [e]):
                            if not mask[idx]:
                                n.style.color = cm_colors["PRED"]
                        
                        # Insert into the overlap stream
                        self._assign_and_insert_into_staff(e, t_staff, b_staff)
                    else:
                        if not mask:
                            e.style.color = cm_colors["PRED"]
                        
                    assert e.offset == time_point
                
                if isinstance(event, tempo.MetronomeMark):
                    t_staff.insert(time_point, event)
                elif isinstance(event, meter.TimeSignature):
                    t_staff.insert(time_point, event)
                    b_staff.insert(time_point, event)      
                
            # Insert any notes in a gt chord that might have been missed (false negatives)
            if not_rest_mask:
                for c, mask in not_rest_mask.items():
                    trimmed_gt = []
                    for idx, n in enumerate(c if c.isChord else [c]):
                        if not mask[idx]:
                            n.style.color = cm_colors["GT"]
                            trimmed_gt.append(n)
                    
                    trimmed_gt = chord.Chord(trimmed_gt)
                    trimmed_gt.offset = time_point
                    trimmed_gt.quarterLength = c.quarterLength
                    assert c.offset == time_point
                    assert trimmed_gt.offset == time_point
                    # Insert into the overlap stream
                    self._assign_and_insert_into_staff(trimmed_gt, t_staff, b_staff)
            
            # Update the ground truth time to see if we missed anything
            gt_time = gt_tree.getPositionAfter(time_point)
            if gt_time is None:
                gt_time = time_point + 1
        
        # If there is more gt stream, insert the rest as false negatives
        gt_time = gt_tree.getPositionAfter(time_point)
        while gt_time is not None:
            for e in gt_tree.elementsStartingAt(gt_time):
                event = e.element
                event.offset = gt_time
                event.style.color = cm_colors["GT"]
                
                if isinstance(e, note.NotRest):
                    self._assign_and_insert_into_staff(event, t_staff, b_staff)             
                elif isinstance(event, tempo.MetronomeMark):
                    t_staff.insert(gt_time, event)
                elif isinstance(event, meter.TimeSignature):
                    if event.offset in processed_time_sig_offsets:
                        continue
                    processed_time_sig_offsets.add(time_point)
                    t_staff.insert(gt_time, event)
                    b_staff.insert(gt_time, event)  
                    
            gt_time = gt_tree.getPositionAfter(gt_time)
                
        
        # ----------------------------- Post process ----------------------------- #
        ref_stream = t_staff if t_staff.highestTime > b_staff.highestTime else b_staff
        
        t_staff.makeVoices(inPlace = True, fillGaps = False)
        b_staff.makeVoices(inPlace = True, fillGaps = False)
        
        self._makeMeasures(t_staff,
                           refStreamOrTimeRange = ref_stream if ref_stream == b_staff else None,
                           inPlace = True)
        t_staff.makeTies(inPlace = True)
        t_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
        t_staff.measure(-1).rightBarline = bar.Barline('final')
        
        self._makeMeasures(b_staff,
                           refStreamOrTimeRange = ref_stream if ref_stream == t_staff else None,
                           inPlace = True)
        b_staff.makeTies(inPlace = True)
        b_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
        b_staff.measure(-1).rightBarline = bar.Barline('final')
        
        overlap_score = stream.Score(id = "overlap")
        
        # Add the staffs to the score
        overlap_score.insert(0, t_staff)
        overlap_score.insert(0, b_staff)
        piano = layout.StaffGroup([t_staff, b_staff], symbol='brace')
        overlap_score.insert(0, piano)

        # Propose key signature
        proposed_key = overlap_score.analyze('key')
        ks = key.KeySignature(proposed_key.sharps)
        
        overlap_score = overlap_score.makeAccidentals(alteredPitches = ks.alteredPitches, 
                                    overrideStatus = True)
        
        overlap_score.parts[0].keySignature = ks
        overlap_score.parts[1].keySignature = ks

        self.plot_pianoroll(overlap_score)
        
        overlap_score.insert(0, metadata.Metadata())
        overlap_score.metadata.title = self.song_name
        overlap_score.metadata.composer = "AMT Model (Red), GT (Blue)"
        
        song_dir = os.path.join(self.output_dir, "overlap", self.song_name)
        overlap_score.write('musicxml', fp=f'{song_dir}.xml')
        self._adjust_voice_numbers(f'{song_dir}.xml')
            
    def create_midi_from_model_events(self, events, bpm_tempo, output_dir='', onset_only=False):
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
        
    def translate_events_to_sheet_music(self, event_sequence: list[tuple[str, int]], output_dir: str) -> stream.Score:
        """
        Args:
            events (list[tuple[str, int]]): example: [('beat', 6), ('onset', 1), ('pitch', 60), ('offset', 0), ('pitch', 60)]
            output_dir (str): Where to save the sheet music

        Returns:
            pd.DataFrame: Dataframe with columns: event, onset and offset.
        """
        
        if len(event_sequence) == 0:
            print(f"Empty event sequence for song: {self.song_name}. Aborting sheet process")
            return
        
        treble_staff = stream.PartStaff(id="TrebleStaff")
        bass_staff = stream.PartStaff(id="BassStaff")
        
        treble_staff.clef = clef.TrebleClef()
        bass_staff.clef = clef.BassClef()
        
        df = self._prepare_data_frame(event_sequence)
        print("Dataframe prepared!")
        
        # Calculate how long each measure should be
        measure_durations = np.diff(df[df['event'].isin(['SOS', 'Downbeat', 'EOS'])]['onset'])

        # Convert time to quarternote length
        df['duration'] = (df['offset'] - df['onset'])
        beats_per_bar = 0
        bar_dur = 0
        i = 0
        for _, row in df.iterrows():

            if row['event'] in ['SOS', 'Downbeat']:
                # If there is no pickup, then don't create a measure for the SOS call
                if measure_durations[i] == 0:
                    i += 1
                    continue

                # If a new time signature is detected
                new_time_signature_detected = ((beats_per_bar != measure_durations[np.max([1, i])]) or (bar_dur != beats_per_bar))
                if new_time_signature_detected:
                    beats_per_bar = measure_durations[np.max([1, i])]
                    
                    multiplier = 1
                    while (beats_per_bar * multiplier) % 1 != 0:
                        multiplier += 1
                        
                        # We are beyond 64 notes. NOTE: This is yet to happen and the consequences are unknown
                        if multiplier > 5:
                            print(f"{self.song_name} has an irregular predicted time signature: {beats_per_bar}, at onset: {row['onset']} " \
                                "We now set it manually to just 4")
                            beats_per_bar = 4
                            multiplier = 1
                    
                    nominator = int(beats_per_bar * multiplier)
                    denominator = 2**(multiplier + 1)
                    
                    ts = meter.TimeSignature(f'{nominator}/{denominator}')
                    ts.barDuration.quarterLength = measure_durations[i]
                    bar_dur = measure_durations[i]
                    
                    treble_staff.insert(row['onset'], ts)
                    bass_staff.insert(row['onset'], ts)
                
                i += 1
        
            elif row['event'] == "Tempo":
                metronome_mark = tempo.MetronomeMark(number = row['offset'])
                metronome_mark.placement = 'above'
                treble_staff.insert(row['onset'], metronome_mark)
                
            elif row['event'] == "EOS":
                pass
                # treble_staff.insert(row['onset'], bar.Barline('final'))
                # bass_staff.insert(row['onset'], bar.Barline('final'))
                
            else:  
                if len(row['event']) > 1:
                    xml_note = chord.Chord(row['event'])
                else:      
                    xml_note = note.Note(row['event'][0])
                
                xml_note.duration = duration.Duration(row['duration'])
                xml_note.offset = row['onset']
                
                if row['duration'] == 0:
                    xml_note = xml_note.getGrace()
                
                self._assign_and_insert_into_staff(xml_note, treble_staff, bass_staff)
                
                # staff.insert(xml_note.offset, xml_note)

        print("Done with adding to the streams. Now we make the score!") 
        # Connect the streams to a score
        score = stream.Score() 
        
        # Post process the two staves
        treble_staff.makeVoices(inPlace = True, fillGaps = False)
        bass_staff.makeVoices(inPlace = True, fillGaps = False)
        
        # Make sure both staves have the same amound of measures and that rests are displayed for all measures
        ref_stream = treble_staff if treble_staff.highestTime > bass_staff.highestTime else bass_staff
        self._makeMeasures(treble_staff,
                           refStreamOrTimeRange = ref_stream if ref_stream == bass_staff else None,
                           inPlace = True)
        treble_staff.makeTies(inPlace = True)
        treble_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
        treble_staff.measure(-1).rightBarline = bar.Barline('final')
        
        self._makeMeasures(bass_staff, 
                           refStreamOrTimeRange = ref_stream if ref_stream == treble_staff else None,
                           inPlace = True)
        bass_staff.makeTies(inPlace = True)
        bass_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
        bass_staff.measure(-1).rightBarline = bar.Barline('final')
        
        # Add the staffs to the score
        score.insert(0, treble_staff)
        score.insert(0, bass_staff)
        piano = layout.StaffGroup([treble_staff, bass_staff], symbol='brace')
        score.insert(0, piano)

        print("We propose a key")
        # Change pitches to the analyzed key 
        # NOTE: If the song changes key throughout, it may screw-up the key analysis
        proposed_key = score.analyze('key')
        ks = key.KeySignature(proposed_key.sharps)

        print("We update the pitches if the key has flats")
        # Only update the pitches if the key signature includes flats
        # if proposed_key.sharps < 0:
        #     score = _update_pitches(score, proposed_key)
        score = score.makeAccidentals(alteredPitches = ks.alteredPitches, 
                                    overrideStatus = True)
        
        # Add the key signature
        score.parts[0].keySignature = ks
        score.parts[1].keySignature = ks

        print("We can show the score and write it to a file")
        # score.show()
        score.insert(0, metadata.Metadata())
        score.metadata.title = self.song_name
        score.metadata.composer = "Arr: AMT Model"
        
        score.write('musicxml', fp=f'{output_dir}.xml')

        # Adjust the voice numbers (MuseScore does not allow 0-numbered voices)
        self._adjust_voice_numbers(f'{output_dir}.xml')
        
        return score

    def _assign_and_insert_into_staff(self, no: note.Note | chord.Chord, t_staff: stream.Stream, b_staff: stream.Stream) -> None:
        # Evenly distribute between treble and bass cleff - could be optimized
        # Split into notes 
        split_chord = no.notes if no.isChord else [no]
        
        bass_notes, treble_notes = [], []
        for n in split_chord:
            if n.octave < 4:
                bass_notes.append(n)
            else:
                treble_notes.append(n)
        
        for c, staff in [(bass_notes, b_staff), (treble_notes, t_staff)]:
            if len(c) == 0:
                continue
            
            if len(c) > 1:
                ch = chord.Chord(c)
                ch.duration = no.duration
                ch.offset = no.offset
                staff.insert(ch.offset, ch)
                
            else:
                c[0].quarterLength = no.quarterLength
                staff.insert(no.offset, c[0])  

    def _adjust_voice_numbers(self, file_path: str) -> None:
        """MuseScore cannot handle non-unique voice numbers or voice 0. 
        This is handled here"""
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Dictionary to keep track of the highest voice number used in each staff
        highest_voice = 1
        voice_mapping = {}
        
        # Find all measure elements to iterate through
        measures = root.findall('.//measure')
        
        for measure in measures:
            for note in measure.findall('.//note'):
                staff = note.find('staff')
                voice = note.find('voice')
                
                if staff is not None and voice is not None:
                    staff_id = int(staff.text)
                    voice_id = int(voice.text)
                    
                    if (staff_id, voice_id) not in voice_mapping.keys():
                        voice_mapping[(staff_id, voice_id)] = highest_voice
                        highest_voice += 1
                    
                    voice.text = str(voice_mapping[(staff_id, voice_id)])
        
        # Delete the current file
        os.remove(file_path)
        
        # Save the adjusted MusicXML file
        tree.write(file_path)

    def _get_note_value(self, pitch: int) -> tuple[str, str]:
        """NOTE: This function could be replaced by
        simply creating the note with the midi value and calling
        .nameWithOctave
        """
        
        # NOTE perhaps less octaves. 
        # a piano has 8 octaves and musescore doesn't even allow that many
        notes = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
        
        octaves = np.arange(11)
        
        return (notes[pitch % 12], str(octaves[pitch // 12] - 1))

    def _makeMeasures(
        self, s: stream.Stream,
        *,
        meterStream=None,
        refStreamOrTimeRange=None,
        searchContext=False,
        innerBarline=None,
        finalBarline='final',
        bestClef=False,
        inPlace=False,
    ):
        """A slightly altered version of Music21's makeMeasures()"""
        
        mStart = None

        # must take a flat representation, as we need to be able to
        # position components, and sub-streams might hide elements that
        # should be contained

        if s.hasPartLikeStreams():
            # can't flatten, because it would destroy parts
            if inPlace:
                returnObj = s
            else:
                returnObj = copy.deepcopy(s)
            for substream in returnObj.getElementsByClass('Stream'):
                self._makeMeasures(substream, meterStream=meterStream,
                                    refStreamOrTimeRange=refStreamOrTimeRange,
                                    searchContext=searchContext,
                                    innerBarline=innerBarline,
                                    finalBarline=finalBarline,
                                    bestClef=bestClef,
                                    inPlace=True,  # copy already made
                                    )
            if inPlace:
                return None
            else:
                return returnObj
        else:
            if s.hasVoices():
                # cannot make flat if there are voices, as would destroy stream partitions
                # parts containing voices are less likely to occur since MIDI parsing changes in v7
                srcObj = s
            else:
                srcObj = s.flatten()
            if not srcObj.isSorted:
                srcObj = srcObj.sorted()
            if not inPlace:
                srcObj = copy.deepcopy(srcObj) # NOTE: copy.deepcopy will reset the barduration attribute
            voiceCount = len(srcObj.voices)

        # may need to look in activeSite if no time signatures are found
        if meterStream is None:
            # get from this Stream, or search the contexts
            meterStream = srcObj.getTimeSignatures(
                returnDefault=True,
                searchContext=False,
                sortByCreationTime=False
            )

        elif isinstance(meterStream, meter.TimeSignature):
            # if meterStream is a TimeSignature, use it
            ts = meterStream
            meterStream = stream.Stream()
            meterStream.insert(0, ts)
        else:  # check that the meterStream is a Stream!
            if not isinstance(meterStream, stream.Stream):
                raise stream.StreamException(
                    'meterStream is neither a Stream nor a TimeSignature!')

        # need a SpannerBundle to store any found spanners and place
        # at the part level
        spannerBundleAccum = spanner.SpannerBundle()

        clefObj = srcObj.clef or srcObj.getContextByClass(clef.Clef)
        if clefObj is None:
            clefObj = srcObj.getElementsByClass(clef.Clef).getElementsByOffset(0).first()
            # only return clefs that have offset = 0.0
            if not clefObj:
                clefObj = clef.bestClef(srcObj, recurse=True)

        # for each element in stream, need to find max and min offset
        # assume that flat/sorted options will be set before processing
        # list of start, start+dur, element
        offsetMapList = srcObj.offsetMap()
        if offsetMapList:
            oMax = max([x.endTime for x in offsetMapList])
        else:
            oMax = 0

        # if a ref stream is provided, get the highest time from there
        # only if it is greater than the highest time yet encountered
        if refStreamOrTimeRange is not None:
            if isinstance(refStreamOrTimeRange, stream.Stream):
                refStreamHighestTime = refStreamOrTimeRange.highestTime
            else:  # assume it's a list
                refStreamHighestTime = max(refStreamOrTimeRange)
            if refStreamHighestTime > oMax:
                oMax = refStreamHighestTime

        # create a stream of measures to contain the offsets range defined
        # create as many measures as needed to fit in oMax
        post = s.__class__()
        post.derivation.origin = s
        post.derivation.method = 'makeMeasures'

        o = 0.0  # initial position of first measure is assumed to be zero
        thisTimeSignature = meterStream.getElementAtOrBefore(o)
        measureCount = 0 if thisTimeSignature.barDuration.quarterLength != thisTimeSignature.numerator * 4 / thisTimeSignature.denominator else 1
        lastTimeSignature = None
        while True:
            # TODO: avoid while True
            m = stream.Measure()
            thisTimeSignature = meterStream.getElementAtOrBefore(o)
            
            if thisTimeSignature is None and lastTimeSignature is None:
                raise stream.StreamException(
                    'failed to find TimeSignature in meterStream; '
                    + 'cannot process Measures')
            if (thisTimeSignature is not lastTimeSignature
                    and thisTimeSignature is not None):
                lastTimeSignature = thisTimeSignature
                m.timeSignature = thisTimeSignature # NOTE: Before it was deepcopied. That would reset the barDuration attribute
            
            if measureCount == 0:
                m.number = measureCount
                m.padAsAnacrusis()
            else:
                m.number = measureCount 
            
            # only add a clef for the first measure when automatically
            # creating Measures; this clef is from getClefs, called above
            if o == 0:
                m.clef = clefObj
                if voiceCount > 0 and s.keySignature is not None:
                    m.insert(0, copy.deepcopy(s.keySignature))
                
            # add voices if necessary (voiceCount > 0)
            for voiceIndex in range(voiceCount):
                v = stream.Voice()
                v.id = voiceIndex  # id is voice index, starting at 0
                m.coreInsert(0, v)
            if voiceCount:
                m.coreElementsChanged()

            # avoid an infinite loop
            if thisTimeSignature.barDuration.quarterLength == 0:
                raise stream.StreamException(
                    f'time signature {thisTimeSignature!r} has no duration')
            
            post.coreInsert(o, m)  # insert measure
            # increment by meter length
            o += thisTimeSignature.barDuration.quarterLength
            if o >= oMax:  # may be zero
                break  # if length of this measure exceeds last offset
            else:
                measureCount += 1

        post.coreElementsChanged()

        # cache information about each measure (we used to do this once per element)
        postLen = len(post)
        postMeasureList = []
        lastTimeSignature = meter.TimeSignature('4/4')  # default.

        for i in range(postLen):
            m = post[i]
            if m.timeSignature is not None:
                lastTimeSignature = m.timeSignature
            # get start and end offsets for each measure
            # seems like should be able to use m.duration.quarterLengths
            mStart = post.elementOffset(m)
            mEnd = mStart + lastTimeSignature.barDuration.quarterLength
            # if elements start fits within this measure, break and use
            # offset cannot start on end
            postMeasureList.append({'measure': m,
                                    'mStart': mStart,
                                    'mEnd': mEnd})

        # populate measures with elements
        for oneOffsetMap in offsetMapList:
            e, start, end, voiceIndex = oneOffsetMap

            # environLocal.printDebug(['makeMeasures()', start, end, e, voiceIndex])
            # iterate through all measures, finding a measure that
            # can contain this element

            # collect all spanners and move to outer Stream
            if isinstance(e, spanner.Spanner):
                spannerBundleAccum.append(e)
                continue

            match = False

            for i in range(postLen):
                postMeasureInfo = postMeasureList[i]
                mStart = postMeasureInfo['mStart']
                mEnd = postMeasureInfo['mEnd']
                m = postMeasureInfo['measure']

                if mStart <= start < mEnd:
                    match = True
                    break

            if not match:
                if start == end == oMax:
                    post.storeAtEnd(e)
                    continue
                else:
                    raise stream.StreamException(
                        f'cannot place element {e} with start/end {start}/{end} within any measures')

            # find offset in the temporal context of this measure
            # i is the index of the measure that this element starts at
            # mStart, mEnd are correct
            oNew = start - mStart  # remove measure offset from element offset

            # insert element at this offset in the measure
            # not copying elements here!

            # in the case of a Clef, and possibly other measure attributes,
            # the element may have already been placed in this measure
            # we need to only exclude elements that are placed in the special
            # first position
            if m.clef is e:
                continue
            # do not accept another time signature at the zero position: this
            # is handled above
            if oNew == 0 and isinstance(e, meter.TimeSignature):
                continue

            # NOTE: cannot use coreInsert here for some reason
            if voiceIndex is None:
                m.insert(oNew, e)
            else:  # insert into voice specified by the voice index
                m.voices[voiceIndex].insert(oNew, e)

        # add found spanners to higher-level; could insert at zero
        for sp in spannerBundleAccum:
            post.append(sp)

        # clean up temporary streams to avoid extra site accumulation
        del srcObj

        # set barlines if necessary
        # lastIndex = len(post.getElementsByClass(stream.Measure)) - 1
        # for i, m in enumerate(post.getElementsByClass(stream.Measure)):
        #     if i != lastIndex:
        #         if innerBarline not in ['regular', None]:
        #             m.rightBarline = innerBarline
        #     else:
        #         if finalBarline not in ['regular', None]:
        #             m.rightBarline = finalBarline
        #     if bestClef:
        #         m.clef = clef.bestClef(m, recurse=True)

        if not inPlace:
            post.setDerivationMethod('makeMeasures', recurse=True)
            return post  # returns a new stream populated w/ new measure streams
        else:  # clear the stored elements list of this Stream and repopulate
            # with Measures created above
            s._elements = []
            s._endElements = []
            s.coreElementsChanged()
            if post.isSorted:
                postSorted = post
            else:
                postSorted = post.sorted()

            for e in postSorted:
                # may need to handle spanners; already have s as site
                s.insert(post.elementOffset(e), e)

    def _prepare_data_frame(self, event_sequence: list[tuple[str, int]]) -> pd.DataFrame:
        """Helper function. Goes through the event sequence creates a dataframe
        with the columns: event, onset and offset"""
        
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
                note_value, octave_value = self._get_note_value(value)
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
                sub_beats = np.arange(0, 1, self.vocab_config['subdivision'])
                tup_beats = np.arange(0, 1, Fraction(self.vocab_config['tuplet_subdivision']))
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
                    new_beat = (value - decrement) * self.vocab_config['subdivision']
                
                beat = new_beat + eos_beats
        
        # Add last downbeat to indicate end-of-score
        df = pd.concat([df, pd.DataFrame([{'event': "EOS", 'onset': beat, 'offset': beat}])], ignore_index=True)
        
        df = df.sort_values(by=['onset', 'offset']).reset_index(drop=True)
        
        # Add relative offsets in reference to each downbeat
        downbeats = df[df['event'] == 'Downbeat']['onset'].values.tolist() + [beat]
        for i in range(len(downbeats) - 1):
            mask = (df['onset'] >= downbeats[i]) & (df['onset'] < downbeats[i+1])
            df.loc[mask, 'rel_onset'] = df.loc[mask, 'onset'] - downbeats[i]
        
        return df

if __name__ == '__main__':    
    # --------------------------------- Collect configs -------------------------------- #
    with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
        pre_configs = yaml.safe_load(f)
    
    with open("Transformer/configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # This is scuffed, but ignore it lolz
    with open("Transformer/configs/vocab_config.yaml", 'r') as f:
        vocab_configs = yaml.safe_load(f)
    
    configs = {"preprocess": pre_configs,
               "vocab": vocab_configs,
               "train": config}
    
    # --------------------------------- Define vocabulary -------------------------------- #
    vocab = Vocabulary(vocab_configs)
    vocab.define_vocabulary(pre_configs['max_beats'])
    
    # ------------------------------- Choose model ------------------------------- #
    model_name = '29-05-24_musescore.pth' 
    
    # ----------------------------- Choose song ----------------------------- #
    new_song_name = "sans"
    
    # ----------------------------- Choose the type of inference ----------------------------- #
    new_song = True
    preprocess = True
    overlap = True
    
    # --------------------------------- Collect directories -------------------------------- #
    output_dir = "inference_songs"
    data_dir= configs['preprocess']['output_dir'] if preprocess else None
    audio_dirs = np.asarray(configs['preprocess']['data_dirs'])
    
    all_paths = [[os.path.exists(os.path.join(aud_path, f"{new_song_name}.{ext}")) for ext in pre_configs['audio_file_extension']] for aud_path in audio_dirs]
    avail_paths = np.any(all_paths, axis = 1)
    if not np.any(avail_paths):
        raise FileNotFoundError(f"{new_song_name} could not be found")
    avail_ext = np.asarray(pre_configs['audio_file_extension'])[np.any(all_paths, axis = 0)][0]
    song_path = os.path.join(audio_dirs[avail_paths][0], f"{new_song_name}.{avail_ext}")
    
    dirs = {"out": output_dir,
            "data": data_dir,
            "audio": song_path}
    
    # ----------------------------- Instantiate inference object ----------------------------- #
    inference = Inference(vocab=vocab, configs = configs, dirs = dirs, 
                          song_name = new_song_name, model_name = model_name)
    
    if new_song:
        print(f"Performing inference on a new song: {new_song_name}")
        init_bpm = 128
        saved_seq = os.path.exists(os.path.join(dirs['out'], "seq_predictions", f"{new_song_name}.npy"))
        score = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)
        
    if preprocess:
        print("Testing if the preprocessing works")
        inference.test_preprocessing()
        
    if overlap:
        print("Performing overlap of the ground truth score and the predicted")
        all_paths = [[os.path.exists(os.path.join(aud_path, f"{new_song_name}.{ext}")) for ext in configs['preprocess']['score_file_extensions']] for aud_path in audio_dirs]
        avail_paths = np.any(all_paths, axis = 1)
        if not np.any(avail_paths):
            raise FileNotFoundError(f"{new_song_name} could not be found")
        avail_ext = np.asarray(configs['preprocess']['score_file_extensions'])[np.any(all_paths, axis = 0)][0]
        song_path = os.path.join(audio_dirs[avail_paths][0], f"{new_song_name}.{avail_ext}")
        
        # Load the ground truth xml
        try:
            gt = converter.parse(song_path)
            gt = gt.makeRests(timeRangeFromBarDuration=True) # Some scores have missing rests and that will completely mess up the expansion
            gt = gt.expandRepeats()
        except:
            raise Exception(f"The ground truth file of {new_song_name} could not be succesfully processed")
        
        # Get predicted score
        init_bpm = gt.metronomeMarkBoundaries()[0][-1].getQuarterBPM()
        saved_seq = os.path.exists(os.path.join(dirs['out'], "seq_predictions", f"{new_song_name}.npy"))
        pred_score = inference.inference(init_bpm = init_bpm, saved_seq = saved_seq, just_save_seq = False)
        
        if pred_score is not None:
            inference.overlap_gt_and_pred(gt, pred_score)