import os
import shutil
from music21 import converter, expressions, repeat, stream

# Define the folder paths
source_folder = '/Users/helenakeitum/Desktop/bachelor_musescore'
fermata_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/fermata_songs'
problem_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/songs_w_problems'
fine_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/songs_fine'

# Create the destination folder if it doesn't exist
os.makedirs(fermata_folder, exist_ok=True)
os.makedirs(problem_folder, exist_ok=True)
os.makedirs(fine_folder, exist_ok=True)

# Function to check for fermatas not in the last measure
def has_fermatas_not_in_last_measure(score):
    # Get the total number of measures
    last_measure_num = max(part.measure(-1).measureNumber for part in score.parts)
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            if measure.measureNumber < last_measure_num:
                for element in measure.notesAndRests:
                    if any(isinstance(expr, expressions.Fermata) for expr in element.expressions):
                        return True
    return False

# Process each MXL or XML file in the source folder
for file_name in os.listdir(source_folder):
    destination_folder = fine_folder
    
    if file_name.endswith(('.mxl', '.xml')):
        file_path = os.path.join(source_folder, file_name)
        
        # Expand repeats
        try:
            score = converter.parse(file_path)
            expanded_score = score.expandRepeats()
        except Exception:
            destination_folder = problem_folder
        
        # Check for fermatas not in the last measure
        if has_fermatas_not_in_last_measure(expanded_score):
            destination_folder = fermata_folder
        
        # Move the music XML/MXL file to the destination folder
        shutil.move(file_path, os.path.join(destination_folder, file_name))
        
        # Check for associated audio files and move them
        base_name, _ = os.path.splitext(file_name)
        for ext in ('.mp3', '.wav'):
            audio_file = base_name + ext
            audio_file_path = os.path.join(source_folder, audio_file)
            if os.path.exists(audio_file_path):
                shutil.move(audio_file_path, os.path.join(destination_folder, audio_file))
    
print("Processing complete.")
