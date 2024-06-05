import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
import yaml
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET

from fractions import Fraction
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from music21 import *
from torch.nn.utils.rnn import pad_sequence

from utils.vocabularies import Vocabulary
from utils.preprocess_song import Song
from utils.model import Transformer

def HelenaMakeMeasures(
    s: stream.Stream,
    *,
    meterStream=None,
    refStreamOrTimeRange=None,
    searchContext=False,
    innerBarline=None,
    finalBarline='final',
    bestClef=False,
    inPlace=False,
):
    
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
            HelenaMakeMeasures(substream, meterStream=meterStream,
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
    measureCount = 0
    lastTimeSignature = None
    while True:
        # TODO: avoid while True
        m = stream.Measure()
        m.number = measureCount + 1
        thisTimeSignature = meterStream.getElementAtOrBefore(o)
        
        if thisTimeSignature is None and lastTimeSignature is None:
            raise stream.StreamException(
                'failed to find TimeSignature in meterStream; '
                + 'cannot process Measures')
        if (thisTimeSignature is not lastTimeSignature
                and thisTimeSignature is not None):
            lastTimeSignature = thisTimeSignature
            m.timeSignature = thisTimeSignature # NOTE: Before it was deepcopied. That would reset the barDuration attribute
            
        # only add a clef for the first measure when automatically
        # creating Measures; this clef is from getClefs, called above
        if measureCount == 0:
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
    lastIndex = len(post.getElementsByClass(stream.Measure)) - 1
    for i, m in enumerate(post.getElementsByClass(stream.Measure)):
        if i != lastIndex:
            if innerBarline not in ['regular', None]:
                m.rightBarline = innerBarline
        else:
            if finalBarline not in ['regular', None]:
                m.rightBarline = finalBarline
        if bestClef:
            m.clef = clef.bestClef(m, recurse=True)

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

def plot_with_custom_colors(m21_stream, alpha=0.5):
    # Flatten the stream to handle all notes and chords
    flat_stream = m21_stream.flatten()
    
    # Extract all notes and chords with their offsets
    elements = [(elem.offset, elem) for elem in flat_stream.notesAndRests]
    
    fig, ax = plt.subplots()

    # Determine the relevant y-axis range
    all_pitches = [p.midi for elem in flat_stream.notesAndRests for p in elem.pitches if isinstance(elem, (note.Note, chord.Chord))]
    if all_pitches:
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
    else:
        min_pitch, max_pitch = 21, 108  # Default range if no notes

    for offset, elem in elements:
        if isinstance(elem, note.Note):
            color = elem.style.color if elem.style.color else 'black'
            ax.broken_barh([(offset, elem.quarterLength)], (elem.pitch.midi - 0.5, 1), facecolors=color, alpha=alpha)
        elif isinstance(elem, chord.Chord):
            for pitch in elem.pitches:
                color = elem.style.color if elem.style.color else 'black'
                ax.broken_barh([(offset, elem.quarterLength)], (pitch.midi - 0.5, 1), facecolors=color, alpha=alpha)
    
    ax.set_xlabel('Time (quarter lengths)')
    ax.set_ylabel('MIDI Pitch')
    
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
    
    plt.show()

def test_preprocessing(data_dir: str):
    songs_dir = os.path.join(data_dir, "spectrograms")
    songs = [os.path.splitext(file)[0] for file in os.listdir(songs_dir)]
    
    for song_name in songs:
        if not os.path.exists(f'{song_name}.mxl'):
            for i in ['train', 'val', 'test']:              
                df = pd.read_csv(os.path.join(data_dir, i, 'labels.csv'))
                
                if (df['song_name'] == song_name).any():
                    break
                
            df = df[df['song_name'] == song_name] # get all segments of the song
            
            # Create a sheet based off of the ground truths
            sequence_events = []
            for idx, row in df.iterrows():
                labels = json.loads(row['labels'])
                events = vocab.translate_sequence_token_to_events(labels)
                sequence_events.extend(events)
            
            translate_events_to_sheet_music(sequence_events, output_dir=f"{song_name}")
        
def inference(init_bpm: int, inference_dir: str, new_song: str, model_name: str, saved_seq: bool, just_save_seq: bool = False):
    os.makedirs(os.path.join(inference_dir, "spectrograms"), exist_ok=True)
        
    new_song_path = os.path.join(inference_dir, new_song) # '/zhome/5d/a/168095/batchelor_amt/test_songs/river.mp3'
    new_song_name = os.path.splitext(new_song)[0]
    if not saved_seq:
        
        # TODO WARNING THIS IS NOT FLEXIBLE
        with open("Transformer/configs/preprocess_config.yaml", 'r') as f:
            preprocess_config = yaml.safe_load(f)
        song = Song(new_song_path, preprocess_config)
        spectrogram = song.compute_spectrogram(save_path=os.path.join(inference_dir, "spectrograms", new_song_name))
        
        model = Transformer(config['n_mel_bins'], tgt_vocab_size, config['d_model'], config['num_heads'], config['num_layers'], config['d_ff'], config['max_seq_length'], config['dropout'], device)
        model.load_state_dict(torch.load(os.path.join('models', model_name), map_location=device))
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
            token_bpm = vocab.vocabulary['tempo'][0].translate_value_to_token(cur_bpm)
            
            # torch.tensor([[1]]) because SOS token is 1 
            output = model.get_sequence_predictions(spec_slice.unsqueeze(0), torch.tensor([[1, token_bpm]], device=device), config['max_seq_length'])
            output = output.squeeze()
            events = vocab.translate_sequence_token_to_events(output.tolist())
            sequence_events.extend(events)
            
            if np.any([event[0] == "tempo" for event in events[2:]]):
                cur_bpm = events[np.where([event[0] == "tempo" for event in events])[0][-1]][1]
        
        os.makedirs(os.path.join(inference_dir, 'seq_predictions'), exist_ok=True)        
        np.save(os.path.join(inference_dir, 'seq_predictions', new_song_name), sequence_events)
    
    else:
        sequence_events = np.load(os.path.join(inference_dir, 'seq_predictions', f'{new_song_name}.npy'))
        
    if not just_save_seq:
        output_dir = os.path.join(inference_dir, 'sheet_predictions')
        os.makedirs(output_dir, exist_ok=True)   
        return translate_events_to_sheet_music(sequence_events, output_dir=os.path.join(output_dir, new_song_name))

def overlap_gt_and_pred(gt_score: stream.Score, pred_score: stream.Score, output_dir: str):
    
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
                    _assign_and_insert_into_staff(gt_e.element, t_staff, b_staff)
                
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
                    _assign_and_insert_into_staff(e, t_staff, b_staff)
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
                assert c.offset == time_point
                assert trimmed_gt.offset == time_point
                # Insert into the overlap stream
                _assign_and_insert_into_staff(trimmed_gt, t_staff, b_staff)
        
        # Update the ground truth time to see if we missed anything
        gt_time = gt_tree.getPositionAfter(time_point)
        if gt_time is None:
            gt_time = time_point + 1
    
    # ----------------------------- Post process ----------------------------- #
    for part in [t_staff, b_staff]:
        part.makeVoices(inPlace = True, fillGaps = False)
        HelenaMakeMeasures(part, inPlace = True)
        part.makeTies(inPlace = True)
        part.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
    
    overlap_score = stream.Score(id = "overlap")
    
    # Add the staffs to the score
    overlap_score.insert(0, t_staff)
    overlap_score.insert(0, b_staff)
    piano = layout.StaffGroup([t_staff, b_staff], symbol='brace')
    overlap_score.insert(0, piano)

    # Propose key signature
    proposed_key = overlap_score.analyze('key')
    if proposed_key.sharps < 0:
        overlap_score = _update_pitches(overlap_score, proposed_key)

    overlap_score.parts[0].keySignature = key.KeySignature(proposed_key.sharps)
    overlap_score.parts[1].keySignature = key.KeySignature(proposed_key.sharps)

    overlap_score.insert(0, metadata.Metadata())
    overlap_score.metadata.title = os.path.basename(output_dir)
    overlap_score.metadata.composer = "AMT Model (Red), GT (Blue)"
    overlap_score.write('musicxml', fp=f'{output_dir}.xml')
    
    _adjust_voice_numbers(f'{output_dir}.xml')
       
        
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
                                    output_dir = "output.xml"):
    """example: [('timeshift', 200), ('onset', 0), ('pitch', 60), ('offset', 100), ('pitch', 62), ('onset', 100)]

    Args:
        events (list[tuple[str, int]]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    treble_staff = stream.PartStaff(id="TrebleStaff")
    bass_staff = stream.PartStaff(id="BassStaff")
    
    treble_staff.clef = clef.TrebleClef()
    bass_staff.clef = clef.BassClef()
    
    df = _prepare_data_frame(event_sequence)
    print("Dataframe prepared!")
    
    # Calculate how long each measure should be
    measure_durations = np.diff(df[df['event'].isin(['SOS', 'Downbeat', 'EOS'])]['onset'])

    # Convert time to quarternote length
    df['duration'] = (df['offset'] - df['onset'])
    beats_per_bar = 0
    bar_dur = 0
    i = 0
    for idx, row in df.iterrows():

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
            treble_staff.append(bar.Barline('final'))
            bass_staff.append(bar.Barline('final'))
            
        else:  
            if len(row['event']) > 1:
                xml_note = chord.Chord(row['event'])
            else:      
                xml_note = note.Note(row['event'][0])
            
            xml_note.duration = duration.Duration(row['duration'])
            xml_note.offset = row['onset']
            
            if row['duration'] == 0:
                xml_note = xml_note.getGrace()
            
            _assign_and_insert_into_staff(xml_note, treble_staff, bass_staff)
            
            # staff.insert(xml_note.offset, xml_note)

    print("Done with adding to the streams. Now we make the score!") 
    # Connect the streams to a score
    score = stream.Score() 
    
    # Post process the two staves
    treble_staff.makeVoices(inPlace = True, fillGaps = False)    
    HelenaMakeMeasures(treble_staff, inPlace = True)
    treble_staff.makeTies(inPlace = True)
    treble_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
    
    bass_staff.makeVoices(inPlace = True, fillGaps = False)
    HelenaMakeMeasures(bass_staff, inPlace = True)
    bass_staff.makeTies(inPlace = True)
    bass_staff.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration = True)
    
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
    score.parts[0].keySignature = key.KeySignature(proposed_key.sharps)
    score.parts[1].keySignature = key.KeySignature(proposed_key.sharps)

    print("We can show the score and write it to a file")
    # score.show()
    score.insert(0, metadata.Metadata())
    score.metadata.title = os.path.basename(output_dir)
    score.metadata.composer = "Arr: AMT Model"
    score.write('musicxml', fp=f'{output_dir}.xml')

    # Adjust the voice numbers (MuseScore does not allow 0-numbered voices)
    _adjust_voice_numbers(f'{output_dir}.xml')
    
    return score

def _assign_and_insert_into_staff(no, t_staff, b_staff):
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
        

def _adjust_voice_numbers(file_path):
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
    
    df = df.sort_values(by=['onset', 'offset']).reset_index(drop=True)
    
    # Add relative offsets in reference to each downbeat
    downbeats = df[df['event'] == 'Downbeat']['onset'].values.tolist() + [beat]
    for i in range(len(downbeats) - 1):
        mask = (df['onset'] >= downbeats[i]) & (df['onset'] < downbeats[i+1])
        df.loc[mask, 'rel_onset'] = df.loc[mask, 'onset'] - downbeats[i]
    
    return df


def _get_note_value(pitch):
    # NOTE perhaps less octaves. 
    # a piano has 8 octaves and musescore doesn't even allow that many
    notes = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 
                6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    octaves = np.arange(11)
    
    return (notes[pitch % 12], str(octaves[pitch // 12] - 1))

if __name__ == '__main__':    
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

    # ----------------------------- For inference ----------------------------- #
    inference_dir = "inference_songs"
    new_song_name = "sans.mp3"
    
    # ----------------------------- For preprocessing ----------------------------- #
    data_dir = "preprocessed_data"
    
    # ------------------------------- Choose model ------------------------------- #
    model_name = '29-05-24_musescore.pth' 
    
    new_song = False
    preprocess = False
    overlap = True
    init_bpm = 128
    
    if new_song:
    
        inference(init_bpm = init_bpm, inference_dir = inference_dir, new_song = new_song_name, model_name = model_name, saved_seq = False, just_save_seq = False)
        
    elif preprocess:
        test_preprocessing(data_dir)
        
    elif overlap:
        os.makedirs(os.path.join(inference_dir, "overlap"), exist_ok=True)
        
        # Load the ground truth xml
        gt_dir = os.path.join(inference_dir, f'{new_song_name.split(".")[0]}.mxl')
        gt = converter.parse(gt_dir)
        gt = gt.expandRepeats()
        
        # Get predicted score
        pred_score = inference(init_bpm = init_bpm, inference_dir = inference_dir, 
                               new_song = new_song_name, model_name = model_name, 
                               saved_seq = True, just_save_seq = False)
        
        output_dir = os.path.join(inference_dir, "overlap", os.path.splitext(new_song_name)[0])
        overlap_gt_and_pred(gt, pred_score, output_dir=output_dir)