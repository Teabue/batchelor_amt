# Preprocessing Steps for the Transformer

## 1. Configure the Config File 
Configure the [preprocessing config file](/Transformer/configs/preprocess_config.yaml)

_insert examplanation of the configs_

## 2.
Insert how to run



## What is done

1. Load the whole audio in
2. Split it to random sequence lengths, up to some max input length
3. Make spectrograms out of the sequences
4. Line up the ground truths (GOOD LUCK WITH THAT LOL)

Now there can be three ways to save the preprocessing:
1. Don't save it at all, assume we have enough memory to keep aaaaaaall the spectrograms in memory LUL
2. (Go-to)Save the spectrogram values (not images!) and a csv depicting the ground truth as well as the path for the saved spectrogram 
3. Save how to reqcreate the sequence spectrogram and the ground truth in a csv and create the spectrograms during run time (This is INCREDIBLY slow even though it saves space and memory.)






