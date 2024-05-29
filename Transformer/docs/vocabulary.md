# Vocabulary 

## Configuration
The [vocabulary config file](/Transformer/configs/vocab_config.yaml) defines all the event types we want and their min and max values.

|                    | **Name**            | **Explanation**                                                                                                                                                                                                                                 |
|--------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Special tokens** | PAD                 | To ensure the same length of spectrogram slices in the dataloader, spectrograms are padded to match the size of the greatest slice in the batch. The padding token informs of this padding                                                      |
|                    | SOS                 | Start-of-sequence token                                                                                                                                                                                                                         |
|                    | EOS                 | End-of-sequence token                                                                                                                                                                                                                           |
|                    | ET                  | It can happen that note onsets will be cut off in the slicing process. These notes should therefore not be marked with an onset in the beginning of the sequence but instead be labelled as a note that is a continuation of a previous segment |
|                    | downbeat            | Marks whenever the first beat of a bar is present                                                                                                                                                                                               |
| **Event types**    | pitch               | Tokens corresponding to the midi pitch numbering system.                                                                                               |
|                    | onset\_offset       | A token where 1 corresponds to the onsetting of subsequent notes and 0 corresponds to offsetting                                                                                                                                    |
|                    | beat                | The chronological order of notes for a specified subdivision of notes and tuplets. Currently uses 32nd notes and 16th tuplets. The value range will be dynamically adjusted according to the min\_beats and max\_beats set in [the preprocessing config file](/Transformer/configs/preprocess_config.yaml)                                                                                                     |
|                    | tempo       | Declares the bpm in quarter notes                                                                                                                                    |
| **Miscellaneous**  | subdivision         | The fraction of a quarter note corresponding to the shortest subdivision of the notes. Currently 32nd notes                                                                                                                                     |
|                    | tuplet\_subdivision | Expresses the fraction of a quarternote corresponding to the minimum subdivision of tuplets. Note that it is given as a string but is converted to a Fraction class object in the code to avoid floating points and rounding errors             |

<mark> NOTE: </mark> beat 0 is currently not implemented, marking the first beat of a segment. This is because it's implied that we always start at a relative beat of 0. The beat tokens should therefore be interpreted as the chronological distance of notes from beat 0. This could be changed later

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

### Future Event Types:
_Event types with (min,max) values:_
- Instrument

_Special characters:_
- UNK - Unknown instrument 


## Rules:

The hierarchy of tokens is given as:

**Initial declarations:**
1. SOS
2. tempo
3. pitches of ET notes (if any)
4. ET (if any)

**Ongoing declarations:**

5. beat
6. tempo
7. downbeat
8. offset
9.  onset
10. pitch

**Termination declarations:**
11. EOS
12. PAD

Do note that if pitches are being offset, those pitch tokens will be appended to the sequence before a possible onset token. 

- When declaring pitches, always order them by pitch value from lowest first to highest last
- If there are both onsets and offsets at a beat, we always offset before we onset
- Always declare already playing pitches first followed by an ET token
- A sequence will be beat shifted to its end duration if there are no notes that are played or if the notes do not play the entire sequence duration


## EXAMPLE:

### Input Segment
```
pitch	onset	offset
20	    --	    1   (onset from previous segment)
19	    --	    2 	(onset from previous segment)	
-1      0       0
-1      1       1
12	    1 	    5/2
-1      2       2
-1      3       3
28	    10/3    31/8
15	    15/4    31/8
```

###  Tokenized:
```
SOS
pitch(19)
pitch(20)
ET()

downbeat          - beat 1
beat(12) 	      - beat 2
downbeat
onset_offset(0)   - offset 
pitch(20)
onset_offset(1)	  - onset
pitch(12)

beat(24)		  - beat 3
downbeat
onset_offset(0)   - offset
pitch(19)

beat(30)		  - beat 3.5 
onset_offset(0)   - offset
pitch(12)

beat(36)
downbeat

beat(40)		  - beat 4 1/3
onset_offset(1)   - onset
pitch(28)

beat(45)          - beat 4 3/4
onset_offset(1)   - onset
pitch(15)

beat(47)          - beat 4 7/8
onset_offset(0)   - offset
pitch(15)
pitch(28)

beat(48)          - shift to end of sequence beat

EOS()
```