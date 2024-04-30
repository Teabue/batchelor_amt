# Preprocessing Steps for the Transformer

## 1. Configure the Config File 
Configure the [preprocessing config file](/Transformer/configs/preprocess_config.yaml)

|                                | **Name**               | **Explanation**                                                                                                                                                                                                                                                                                                                      |
|--------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Splits**                     | train                  | % of data for training                                                                                                                                                                                                                                                                                                               |
|                                | val                    | % of data for validation                                                                                                                                                                                                                                                                                                             |
|                                | test                   | % of data for testing                                                                                                                                                                                                                                                                                                                |
|                                | seed                   | Seed for reproducibility                                                                                                                                                                                                                                                                                                             |
|                                | dataset                | Which dataset to use. Currently allows Maestro and MuseScore. This will determine how the data is processed later as some functionalities are only present in certain datasets. Maestro allows for MIDI-like prediction (shifts tokens in time) while MuseScore allows for XML-like predictions (shifts tokens in quarternote beats) |
| **Spectrogram configurations** | max\_sequence\_length  | Only relevant when using the Maestro dataset. Determines the maximum allowed spectrogram sequence length as input to the model                                                                                                                                                                                                       |
|                                | preprocess_method      | Hyperparameter. Determines which spectrogram algorithm to use                                                                                                                                                                                                                                                                        |
|                                | h\_bars                | Hyperparameter. Only relevant when using the MuseScore dataset. Determines the number of bars a spectrogram slice contains                                                                                                                                                                                                           |
|                                | sr                     | Sampling rate                                                                                                                                                                                                                                                                                                                        |
|                                | n\_fft                 | Number of samples to use for each Fourier Transform                                                                                                                                                                                                                                                                                  |
|                                | hop\_length            | The number of samples that overlap between frames                                                                                                                                                                                                                                                                                    |
| **Directories**                | data\_dirs             | Location of dataset(s) including the audio and ground truth annotations                                                                                                                                                                                                                                                              |
|                                | audio\_file\_extension | The allowed file extensions for the audio data                                                                                                                                                                                                                                                                                       |
|                                | gt\_file\_extensions   | The allowed file extensions for the ground truth data. Be wary that depending on which dataset is chosen, certain file extensions will raise an error i.e using midi for MuseScore data                                                                                                                                              |
|                                | output\_dir            | The location of where to store the preprocessed spectrograms and the training, validation and testing ground truth labels for each spectrogram slice                                                                                                                                                                                 |
| **Miscellaneous**              | num\_workers           | The number of workers to use                                                                                                                                                                                                                                                                                                         |


## 2.
To run the preprocessing, run the [main data preprocessor file](/Transformer/data_preprocessor.py)


## What is done

1. Load the audio and split the audio data into training, testing and validation
2. Compute the spectrogram of the audio using the spectrogram configurations
3. Compute onsets and offsets in the audio along with downbeats
    1. If using the Maestro dataset, the onsets and offsets will be in ms
    2. If using the MuseScore dataset, the onsets and offsets will be in quarternote beats
4. Slice the dataframe onsets/offsets/downbeats into segments
    1. If using the Maestro dataset, segments will be split it to random sequence lengths, up to some max input length
    2. If using the MuseScore dataset, segments will be split into a predefined number of bars. If there is a remainder of bars less than the slicing length, these will be omitted 
5. Translate the ground truths to tokens and save everything in a csv file
    1. The csv file will contain the name of the song, the start and end indices of the spectrogram segment as well as the ground truth tokens


<mark> **Note that there can be three ways to save the preprocessing** </mark>:
1. Don't save it at all, assume we have enough memory to keep all the spectrograms in memory LUL
2. (Go-to) Save the spectrogram values (not images!) and a csv depicting the ground truth as well as the path for the saved spectrogram 
3. Save how to recreate the sequence spectrogram and the ground truth in a csv and create the spectrograms during run time (This is INCREDIBLY slow even though it saves space and memory.)