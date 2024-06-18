import os
import mido
import pandas as pd
import glob
from multiprocessing import Pool
from tqdm import tqdm


def process_wav_files(args):
    dir_path, wav_files = args
    for wav_file in wav_files:
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        midi_file = os.path.join(dir_path, base_name + '.midi')
        mid_file = os.path.join(dir_path, base_name + '.mid')
        
        if os.path.exists(midi_file):
            create_maestro_notation(midi_file, base_name)
        elif os.path.exists(mid_file):
            create_maestro_notation(mid_file, base_name)
            
def create_maestro_notation(song_path, song_name, sanity_check=True):
    if os.path.exists(os.path.join(os.path.dirname(song_path), song_name + '.midi')):
        midi_path = os.path.join(os.path.dirname(song_path), song_name + '.midi')
    elif os.path.exists(os.path.join(os.path.dirname(song_path), song_name + '.mid')):
        midi_path = os.path.join(os.path.dirname(song_path), song_name + '.mid')
    else:
        raise FileNotFoundError(f"No .midi or .mid file found for {song_name}")
    
    # Load midi file to get tempo
    midi_data = mido.MidiFile(midi_path)
    cur_tempo = None
    midi_msgs = mido.merge_tracks(midi_data.tracks)

    df = pd.DataFrame(columns=['MidiPitch', 'OnsetTime', 'OffsetTime']) # midi_pitch, onset time and offset time in seconds
    cur_time = 0
    
    for msg in midi_msgs:
        if msg.type == 'set_tempo':
            cur_tempo = msg.tempo
            continue

        if (cur_tempo == None and msg.time != 0) or cur_tempo != None:
            if (cur_tempo == None and msg.time != 0):
                with open('MIDI_WO_TEMPO', 'a') as f:
                    f.write(song_name + '\n')
                cur_tempo = 500000 # Set tempo to 120 bpm
    
            cur_time += mido.tick2second(msg.time, ticks_per_beat=midi_data.ticks_per_beat, tempo=cur_tempo)
            if msg.type == 'note_on' and msg.velocity > 0:
                df = pd.concat([df, pd.DataFrame([{'MidiPitch': msg.note, 'OnsetTime': cur_time}])], ignore_index=True)

            # For some god awful reason, the Maestro dataset don't use note_off events, but note_on events with velocity 0 >:((
            elif (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
                # fill out the note_off event
                df.loc[(df['MidiPitch'] == msg.note) & (df['OffsetTime'].isnull()), 'OffsetTime'] = cur_time
        
    df.to_csv(os.path.join(os.path.dirname(song_path), song_name + '.txt'), sep=' ', index=False)
    
if __name__ == '__main__':
    data_dirs = [
        '/work3/s214629/data/asap_generated',
        '/work3/s214629/data/Lieder_generated',
        '/work3/s214629/data/music21_generated'
    ]
    num_processes = 20  # Adjust based on your system's capabilities

    for dir_path in data_dirs:
        wav_files = glob.glob(os.path.join(dir_path, '*.wav'))
        # Split wav_files into chunks for each process
        chunks = [wav_files[i::num_processes] for i in range(num_processes)]
        
        with Pool(num_processes) as pool:
            list(tqdm(pool.imap(process_wav_files, [(dir_path, chunk) for chunk in chunks]), total=len(chunks)))