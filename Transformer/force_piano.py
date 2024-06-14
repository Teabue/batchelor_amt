import os
import yaml

from collections import defaultdict
from music21 import converter, instrument, stream

# Load preprocessing configs
with open('Transformer/configs/preprocess_config.yaml', 'r') as f:
    preproc_config = yaml.safe_load(f)

# Find all the xml files in each dataset
for dataset in preproc_config['data_dirs']:
    for file in os.listdir(dataset):
        if file.endswith(tuple(preproc_config['score_file_extensions'])):
            
            # Load ground truth score
            score_path = os.path.join(dataset, file)
            score = converter.parse(score_path)
            score.id = "Original"
            
            # Check if piano or organ is already in the score
            all_instruments = list(score.getInstruments())
            piano_present = instrument.Piano() in all_instruments
            all_parts = list(score.parts)
            
            # Purge all other instruments
            for part in all_parts:
                
                if part.partName == "Percussion":
                    score.remove(part)
                
                elif part.partName != 'Piano':
                    if not piano_present:
                        # Change to piano
                        wrong_instrument = part.getInstrument()
                        part.remove(wrong_instrument)
                        part.insert(0, instrument.Piano())
                    else:
                        score.remove(part)
            
            # Remove any duplicate notes that may be present
            pitches_at_onset = defaultdict(list)
            notes_to_remove = []
            for note in score.recurse().notes:
                onset = note.getOffsetInHierarchy(score)
                for p in note.pitches:
                    if p in pitches_at_onset[onset]:
                        notes_to_remove.append(note)
                    else:
                        pitches_at_onset[onset].append(p.midi)
            
            score.remove(notes_to_remove, recurse = True)
            score.makeRests(fillGaps = True, inPlace = True, timeRangeFromBarDuration=True)
            
            # Overwrite to a new file
            score.write('xml', fp = score_path)