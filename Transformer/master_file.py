import os

# Run BeatTrack model
model = "BeatTrack"
for fourier in ['stft', 'cqt', 'logmel']:
    for loss in ['ce', 'cl']:
        
        # Preprocess
        command = f'python Transformer/data_preprocessor.py --model {model} --fourier {fourier}'
        os.system(command)
        
        # Train
        command = f'python Transformer/train.py --loss {loss}'
        os.system(command)
        
        # Run inference
        if model == 'BeatTrack':
            command = f'python Transformer/inference.py --fourier {fourier} --loss {loss}'
            os.system(command)
        elif model == 'TimeShift':
            command = f'python Transformer/midi_inference.py --fourier {fourier}'
            os.system(command)
        else:
            raise ValueError("Invalid model")