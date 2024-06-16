import os

# Run BeatTrack model
model = "BeatTrack"
for fourier in ['stft', 'cqt', 'logmel']:
    for loss in ['ce', 'cl']:
        
        # Preprocess
        command = f'python Transformer/data_preprocessor.py --{model} --{fourier}'
        os.system(command)
        
        # Train
        command = f'python Transformer/train.py --{loss}'
        os.system(command)
        
        # Run inference
        command = f'python Transformer/inference.py --{fourier} --{loss}'
        os.system(command)