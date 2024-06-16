import os

# Run BeatTrack model
model = "BeatTrack"
for fourier in ['stft', 'cqt', 'logmel']:
    for loss in ['ce', 'cl']:
        
        # Run inference
        if model == 'BeatTrack':
            command = f'python Transformer/inference.py --fourier {fourier} --loss {loss}'
            os.system(command)
        elif model == 'TimeShift':
            command = f'python Transformer/midi_inference.py --fourier {fourier}'
            os.system(command)
        else:
            raise ValueError("Invalid model")