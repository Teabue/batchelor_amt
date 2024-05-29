import os
import shutil
from music21 import converter, expressions, note
import numpy as np

def fermata_check(score):
    """Function to check for fermatas not in the last measure"""
    
    # Get the total number of measures
    last_measure_num = max(part.measure(-1).measureNumber for part in score.parts)
    
    for part in score.parts:
        for element in part.flatten().getElementsByClass(note.GeneralNote):
            for expr in element.expressions:
                if isinstance(expr, expressions.Fermata) and element.measureNumber < last_measure_num:
                    return True, element.offset
        
    return False, score.highestTime

if __name__ == "__main__":
    # Define the folder paths
    source_folder = '/Users/helenakeitum/Desktop/bachelor_musescore'
    fermata_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/fermata_songs'
    problem_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/songs_w_problems'
    fine_folder = '/Users/helenakeitum/Desktop/bachelor_musescore/songs_fine'

    # Create the destination folder if it doesn't exist
    os.makedirs(fermata_folder, exist_ok=True)
    os.makedirs(problem_folder, exist_ok=True)
    os.makedirs(fine_folder, exist_ok=True)

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
            if fermata_check(expanded_score)[0]:
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
