
# VOCABULARY

## Configuration
The [vocabulary config file](/Transformer/configs/vocab_config.yaml) defines all the event types we want and their min and max values.

Note: Time shift is a bit special, it's easier to work with seconds and more flexible, so the min max values will automatically be determined through:
- STEPS_PER_SECOND defaults to 100, which means that each time shift will shift 10 ms
- MAX_SHIFT_SECONDS defaults to 10 


## Tokenization 

The decoder outputs will be an integer token for each word. The meaning of the integer depends on the [vocabulary config file](/Transformer/configs/vocab_config.yaml)

- The first token integers will be used for special characters in the order they're defined in the [config file](/Transformer/configs/vocab_config.yaml) 
- Then time shift tokens will range from min to a configged max sequence length time or we could do `(min, MAX_SHIFT_SECONDS * STEPS_PER_SECOND)`, depends on what the max sequence length is
- Then the event types in order. 

### Example:
A config file of:
```
SpecialTokens:
  - PAD
  - EOS
  - ET

EventTypes: 
  Pitch: (0,127)

```
Gives:

```
0: EOS
1: ET
2-129: Pitch
```
Where for example a token integer of value 4 means Pitch(3)

### Current Event Types:
_Event types with (min,max) values:_
- Time_shift  - shifts the time forward
- Pitch - Pitch of sound
- Onset_offset - whether the subsequent pitches will be onset or offset (0: Offset, 1: Onset)

_Special characters with single values:_
- EOS - end of sequence
- ET - end tie; used to indicate it's the end of the declaration of already active pitches.


### Future Event Types:
_Event types with (min,max) values:_
- Instrument

_Special characters:_
- UNK - Unknown instrument 


## Rules:
- When doing pitches, always order them by pitch value from lowest first to highest last
- If there are both onsets and offsets for one time shift, then always keep whatever mode was on, then change the mode
- Always declare already playing pitches first then use ET token
    - Do the normal time shifting and append the pitch tokens. (ignore all tokens in the sequence until this part has been handled) and the normal token-timeshifting will assume it starts from 0 after ET has been done.
        - we could just add a note_offset thing to make stuff easier to parse in inference
- Time shift comes BEFORE onsets and offset tokens
- A sequence will be time shifted to its end duration if there are no notes that are played.



## EXAMPLE:

### Input Segment
```
Pitch	onset	offset
20	--	3.5s   (onset from previous segment)
19	--	1s	(onset from previous segment)	
12	3s	4s
15	3s	3.5s
28	4s	4.5s
```

###  Tokenized:
```
Pitch(19)
Pitch(20)
ET()

Time_shift(100) 	- t: 1s
Onset_offset(0)		- offset 
Pitch(19)

Time_shift(200)		- t: 3s
Onset_offset(1)		- onset
Pitch(12)
Pitch(15)

Time_shift(50)		- t: 3.5s
Onset_offset(0)		- offset
Pitch(15)
Pitch(20)

Time_shift(50)		- t: 4s
Pitch(12)
Onset_offset(0)		- onset
Pitch(28)

Time_shift(50)		- t: 4.5s
Onset_offset(0)
Pitch(28)

EOS()
```

