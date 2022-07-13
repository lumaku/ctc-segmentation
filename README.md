# CTC segmentation

<!-- Badges -->
[![build status](https://github.com/lumaku/ctc-segmentation/actions/workflows/python-package.yml/badge.svg)](https://github.com/lumaku/ctc-segmentation/actions/workflows/python-package.yml)
[![version](https://img.shields.io/pypi/v/ctc-segmentation)](https://pypi.org/project/ctc-segmentation/)
[![AUR](https://img.shields.io/aur/version/python-ctc-segmentation-git)](https://aur.archlinux.org/packages/python-ctc-segmentation-git)
[![downloads](https://img.shields.io/pypi/dm/ctc-segmentation)](https://pypi.org/project/ctc-segmentation/)

CTC segmentation can be used to find utterance alignments within large audio files.

* This repository contains the `ctc-segmentation` python package.
* A description of the algorithm is in the CTC segmentation paper (on [Springer Link](https://link.springer.com/chapter/10.1007%2F978-3-030-60276-5_27), on [ArXiv](https://arxiv.org/abs/2007.09127))


# Usage

The CTC segmentation package is not standalone, as it needs a neural network with CTC output. It is integrated in these frameworks:

* In ESPnet 1 as corpus recipe: [Alignment script](https://github.com/espnet/espnet/blob/master/espnet/bin/asr_align.py), [Example recipe](https://github.com/espnet/espnet/tree/master/egs/tedlium2/align1), [Demo](https://github.com/espnet/espnet#ctc-segmentation-demo )
* In ESPnet 2, as script or directly as python interface: [Alignment script](https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_align.py), [Demo](https://github.com/espnet/espnet#ctc-segmentation-demo )
* In Nvidia NeMo as dataset creation tool: [Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/ctc_segmentation.html), [Example](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb)
* In Speechbrain, as python interface: [Alignment module](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/alignment/ctc_segmentation.py), [Examples](https://gist.github.com/lumaku/75eca1c86d9467a54888d149dc7b84f1)

It can also be used with other frameworks:

<details><summary>Wav2vec2 example code</summary><div>

```python
import torch
import numpy as np
from typing import List
import ctc_segmentation
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

# load model, processor and tokenizer
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# load dummy dataset and read soundfiles
SAMPLERATE = 16000
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]
transcripts = ["A MAN SAID TO THE UNIVERSE", "SIR I EXIST"]

def align_with_transcript(
    audio : np.ndarray,
    transcripts : List[str],
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
    
    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]
    
    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = tokenizer(transcript.replace("\n"," ").lower())['input_ids']
        tok_ids = np.array(tok_ids,dtype=np.int)
        tokens.append(tok_ids[tok_ids != unk_id])
    
    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
    return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : p[2]} for t,p in zip(transcripts, segments)]
    
def get_word_timestamps(
    audio : np.ndarray,
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)
    
    # Split the transcription into words
    words = pred_transcript.split(" ")
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    return [{"text" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]

print(align_with_transcript(audio,transcripts))
# [{'text': 'A MAN SAID TO THE UNIVERSE', 'start': 0.08124999999999993, 'end': 2.034375, 'conf': 0.0}, 
#  {'text': 'SIR I EXIST', 'start': 2.3260775862068965, 'end': 4.078771551724138, 'conf': 0.0}]

print(get_word_timestamps(audio))
# [{'text': 'a', 'start': 0.08124999999999993, 'end': 0.5912715517241378, 'conf': 0.9999501323699951}, 
# {'text': 'man', 'start': 0.5912715517241378, 'end': 0.9219827586206896, 'conf': 0.9409108982174931}, 
# {'text': 'said', 'start': 0.9219827586206896, 'end': 1.2326508620689656, 'conf': 0.7700278702302796}, 
# {'text': 'to', 'start': 1.2326508620689656, 'end': 1.3529094827586206, 'conf': 0.5094435178226225}, 
# {'text': 'the', 'start': 1.3529094827586206, 'end': 1.4831896551724135, 'conf': 0.4580493446392211}, 
# {'text': 'universe', 'start': 1.4831896551724135, 'end': 2.034375, 'conf': 0.9285054256219009}, 
# {'text': 'sir', 'start': 2.3260775862068965, 'end': 3.036530172413793, 'conf': 0.0}, 
# {'text': 'i', 'start': 3.036530172413793, 'end': 3.347198275862069, 'conf': 0.7995760873559864}, 
# {'text': 'exist', 'start': 3.347198275862069, 'end': 4.078771551724138, 'conf': 0.0}]
```

</div></details>



# Installation

* With `pip`:
```sh
pip install ctc-segmentation
```

* From the Arch Linux AUR as `python-ctc-segmentation-git` using your favourite AUR helper.

* From source:
```sh
git clone https://github.com/lumaku/ctc-segmentation
cd ctc-segmentation
cythonize -3 ctc_segmentation/ctc_segmentation_dyn.pyx
python setup.py build
python setup.py install --optimize=1 --skip-build
```

# How it works

### 1. Forward propagation

Character probabilites from each time step are obtained from a CTC-based network.
With these, transition probabilities are mapped into a trellis diagram.
To account for preambles or unrelated segments in audio files, the transition cost are set to zero for the start-of-sentence or blank token.

![Forward trellis](doc/1_forward.png)

### 2. Backtracking

Starting from the time step with the highest probability for the last character, backtracking determines the most probable path of characters through all time steps.

![Backward path](doc/2_backtracking.png)

### 3. Confidence score

As this method generates a probability for each aligned character, a confidence score for each utterance can be derived.
For example, if a word within an utterance is missing, this value is low.

![Confidence score](doc/3_scoring.png)

The confidence score helps to detect and filter-out bad utterances.


# Parameters

There are several notable parameters to adjust the working of the algorithm that can be found in the class `CtcSegmentationParameters`:


### Data preparation parameters

* Localization: The character set is taken from the model dict, i.e., usually are generated with SentencePiece. An ASR model trained in the corresponding language and character set is needed. For asian languages, no changes to the CTC segmentation parameters should be necessary. One exception: If the character set contains any punctuation characters, "#", or the Greek char "ε", adapt the setting in an instance of `CtcSegmentationParameters` in `segmentation.py`.

* `CtcSegmentationParameters` includes a blank character. Copy over the Blank character from the dictionary to the configuration, if in the model dictionary e.g. "\<blank>" instead of the default "_" is used. If the Blank in the configuration and in the dictionary mismatch, the algorithm raises an IndexError at backtracking.

* If `replace_spaces_with_blanks` is True, then spaces in the ground truth sequence are replaces by blanks. This option is enabled by default and improves compability with dictionaries with unknown space characters.

### Alignment parameters

* `min_window_size`: Minimum window size considered for a single utterance. The current default value should be OK in most cases.

* To align utterances with longer unkown audio sections between them, use `blank_transition_cost_zero` (default: False). With this option, the stay transition in the blank state is free. A transition to the next character is only consumed if the probability to switch is higher. In this way, more time steps can be skipped between utterances. Caution: in combination with `replace_spaces_with_blanks == True`, this may lead to misaligned segments.

### Time stamp parameters

Directly set the parameter `index_duration` to give the corresponding time duration of one CTC output index (in seconds).

**Example:** For a given sample rate, say, 16kHz, `fs=16000`. Then, how many sample points correspond to one ctc output index? In some ASR systems, this can be calculated from the hop length of the windowing times encoder subsampling factor. For example, if the hop length of the frontend windowing is 128, and the subsampling factor in the encoder is 4, totalling 512 sample points for one CTC index. Then `index_duration = 512 / 16000`.

**Note:** In earlier versions, `index_duration` was not used and the time stamps were determined from the values of `subsampling_factor` and `frame_duration_ms`. To derive `index_duration` from these values, calculate`frame_duration_ms * subsampling_factor / 1000`.

### Confidence score parameters

Character probabilities over each L frames are accumulated to calculate the confidence score. The L value can be adapted with with `score_min_mean_over_L` . A lower L makes the score more sensitive to error in the transcription, but also errors in the ASR model.


# Toolkit Integration

CTC segmentation requires CTC activations of an already trained CTC-based network. Example code can be found in the alignment scripts `asr_align.py` of ESPnet 1 or ESpnet 2.

### Steps to alignment for regular ASR

1. First, the ground truth text need to be converted into a matrix: Use `prepare_token_list` from a ground truth sequence that was already coverted to a sequence of tokens (recommended). Alternatively, use `prepare_text` on raw text, this method filters characters not in the dictionary and can break longer tokens into smaller tokens.
2. `ctc_segmentation` computes character-wise alignments from the CTC log posterior probabilites.
3. `determine_utterance_segments` converts char-wise alignments to utterance-wise alignments.
4. In a post-processing step, segments may be filtered by their confidence value.

### Steps to alignment for different use-cases

Sometimes the ground truth data is not text, but a sequence of tokens, or a list of protein segments.
In this case, use either `prepare_token_list` or replace it with a function that suits better for your data.
For examples, see the `prepare_*` functions in `ctc_segmentation.py`, or the example included in the NeMo toolkit.

### Segments clean-up

Segments that were written to a `segments` file can be filtered using the confidence score. This is the minium confidence score in log space as described in the paper. 

Utterances with a low confidence score are discarded in a data clean-up. This parameter may need adjustment depending on dataset, ASR model and used text conversion.

```bash
min_confidence_score=1.5
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${unfiltered} > ${filtered}
```


# FAQ

* *How do I split a large text into multiple utterances?* This can be done automatically, e.g. in our paper, the text of this book/chapter was split into utterances at sentence endings to derive utterances.

* *What if there are unrelated parts within the audio file without transcription?* Unrelated segments can be skipped with the `gratis_blank` parameter. Larger unrelated segments may deteriorate the results, try to increase the minimum window size. Partially repeating segments have a high chance to disarrange the alignments, remove them if possible. These segments can be detected with the confidence score. Use the `state_list` to see how well the unrelated part was "ignored".

* *How fast is CTC segmentation?* On a modern computer (64GB RAM, Nvidia RTX 2080 ti, AMD Ryzen 7 2700X), it takes around 400 ms to align 500s of audio with tokenized text and the default window size (8000~250s). In comparison, inference of CTC posteriors on CPU takes some time from 4s to 20s; GPU inference takes roughly 500 - 1000 ms, but often fails at such long audio files because of excessive memory consumption (tested with Transformer model on Espnet 2; this strongly depends on model architecture and used toolkit). A few factors influence CTC segmentation speed: Window size, length of audio, length of text, how well the text fits to audio and the preprocessing function. Aligning from tokenized text is faster because the alignment with `prepare_text` additionally includes transition probabilities from partial tokens; this increases the complexity by the length of the longest token in the character dictionary (including the blank token).

* *The inference of the model is too slow and uses too much memory.* Use RNN-based ASR networks that grow linearly in complexity with longer audio files. Inference complexity increases on long audio files quadratically for Transformer-based architectures. To solve this, it is possible to partition the audio into several parts. CTC segmentation includes an example partitioning function in `ctc_segmentation.get_partitions`. [Example Code in JtubeSpeech](https://github.com/sarulab-speech/jtubespeech/blob/a16cffc1d38ac23f43a230c68ea0927ed9b6ea9f/scripts/align.py#L339-L387)

* *How can I improve the alignment speed of CTC segmentation?* The alignment algorithm is not parallelizable for batch processing, so use a CPU with a good single-thread performance. It's possible to align multiple files in parallel, if the computer has enough temporary memory. The alignment is faster with shorter max token length, if text is aligned - or directly align from a token list.

* *How do I get word-based alignments instead of full utterance segments?* Use an ASR model with character tokens to improve the time-resolution. Then handle each word as single utterance.

* *How can I improve the accuracy of the generated alignments?* Be aware that depending on the ASR performance of network and other factors, CTC activations are not always accurate, sometimes shifted by a few frames. To get a better time resolution, use a dictionary with characters! Also, the `prepare_text` function tries to break down long tokens into smaller tokens.

* *What is the difference between `prepare_token_list` and `prepare_text`?* Explained in examples:

<details><summary>Example for `prepare_token_list`</summary><div>

Let's say we have a text `text = ["cat"]` and a dictionary that includes the word cat as well as its parts: `char_list = ["•", "UNK", "a", "c", "t", "cat"]`.
The "tokenize" method that uses the `preprocess_fn` will produce:

```python
text = ["cat"]
char_list = ["•", "UNK", "a", "c", "t", "cat"]
token_list = [tokenize(utt) for utt in text]
token_list
# [array([5])]
ground_truth_mat, utt_begin_indices = prepare_token_list(config, text)
ground_truth_mat
# array([[-1],
#        [ 0],
#        [ 5],
#        [ 0]])
```

</div></details>

<details><summary>Example for `prepare_text`</summary><div>
Toy example:

```python
text = ["cat"]
char_list = ["•", "UNK", "a", "c", "t", "cat"]
ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
# array([[-1, -1, -1],
#        [ 0, -1, -1],
#        [ 3, -1, -1],
#        [ 2, -1, -1],
#        [ 4, -1,  5],
#        [ 0, -1, -1]])
```

Here, the partial characters are detected (3,2,4), as well as the full "cat" token (5).
This is done to have a better time resolution for the alignment.

Full example with a bpe 500 model char list from Tedlium 2:

```python

from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import prepare_text

char_list = [ "<unk>", "'", "a", "ab", "able", "ace", "ach", "ack",
"act","ad","ag", "age", "ain", "al", "alk", "all", "ally", "am",
"ame", "an","and", "ans", "ant", "ap", "ar", "ard", "are",
"art","as", "ase","ass", "ast", "at", "ate", "ated", "ater", "ation",
"ations", "ause","ay", "b", "ber", "ble", "c", "ce", "cent",
"ces","ch", "ci", "ck","co", "ct", "d", "de", "du", "e", "ear",
"ect", "ed", "een", "el","ell", "em", "en", "ence", "ens",
"ent","enty", "ep", "er", "ere","ers", "es", "ess", "est", "et", "f",
"fe", "ff", "g", "ge", "gh","ght", "h", "her", "hing", "ht",
"i","ia", "ial", "ib", "ic","ical", "ice", "ich", "ict", "id", "ide",
"ie", "ies", "if","iff","ig", "ight", "ign", "il", "ild", "ill","im",
"in", "ind","ine", "ing", "ink", "int", "ion", "ions", "ip",
"ir","ire","is","ish", "ist", "it", "ite", "ith", "itt", "ittle",
"ity", "iv","ive", "ix", "iz", "j", "k", "ke", "king", "l",
"ld","le","ll","ly", "m", "ment", "ms", "n", "nd", "nder", "nt", "o",
"od","ody", "og", "ol", "olog", "om", "ome", "on",
"one","ong","oo","ood", "ook", "op", "or", "ore", "orm", "ort",
"ory", "os","ose", "ot", "other", "ou", "ould", "ound",
"ount","our","ous","ousand", "out", "ow", "own", "p", "ph", "ple",
"pp", "pt","q", "qu", "r", "ra", "rain", "re", "reat", "red",
"ree","res","ro","rou", "rough", "round", "ru", "ry", "s", "se",
"sel", "so","st","t", "ter", "th", "ther", "ty", "u", "ually",
"ud","ue", "ul","ult", "um", "un", "und", "ur", "ure", "us", "use",
"ust", "ut","v","ve", "vel", "ven", "ver", "very", "ves", "ving","w",
"way","x", "y", "z", "ăť", "ō", "▁", "▁a", "▁ab",
"▁about","▁ac","▁act","▁actually", "▁ad", "▁af", "▁ag", "▁al",
"▁all", "▁also","▁am", "▁an", "▁and", "▁any", "▁ar",
"▁are","▁around", "▁as","▁at","▁b", "▁back", "▁be", "▁bec",
"▁because", "▁been", "▁being","▁bet", "▁bl", "▁br", "▁bu",
"▁but","▁by", "▁c", "▁call","▁can","▁ch", "▁chan", "▁cl", "▁co",
"▁com", "▁comm", "▁comp","▁con", "▁cont", "▁could", "▁d",
"▁day","▁de", "▁des","▁did","▁diff", "▁differe", "▁different",
"▁dis", "▁do", "▁does","▁don", "▁down", "▁e", "▁en",
"▁even","▁every", "▁ex", "▁exp","▁f","▁fe", "▁fir", "▁first",
"▁five", "▁for", "▁fr", "▁from", "▁g","▁get", "▁go",
"▁going","▁good", "▁got", "▁h", "▁ha","▁had","▁happ", "▁has",
"▁have", "▁he", "▁her", "▁here", "▁his","▁how", "▁hum",
"▁hundred","▁i", "▁ide", "▁if", "▁im", "▁imp","▁in","▁ind", "▁int",
"▁inter", "▁into", "▁is", "▁it", "▁j", "▁just","▁k","▁kind",
"▁kn","▁know", "▁l", "▁le", "▁let", "▁li", "▁life","▁like",
"▁little", "▁lo", "▁look", "▁lot", "▁m",
"▁ma","▁make","▁man","▁many", "▁may", "▁me", "▁mo", "▁more",
"▁most","▁mu", "▁much", "▁my", "▁n", "▁ne", "▁need", "▁new",
"▁no","▁not","▁now", "▁o", "▁of", "▁on", "▁one","▁only", "▁or",
"▁other", "▁our","▁out", "▁over", "▁p", "▁part", "▁pe",
"▁peop","▁people","▁per","▁ph", "▁pl", "▁po", "▁pr", "▁pre", "▁pro",
"▁put", "▁qu", "▁r","▁re", "▁real", "▁really", "▁res",
"▁right","▁ro", "▁s","▁sa","▁said", "▁say", "▁sc", "▁se", "▁see",
"▁sh", "▁she", "▁show", "▁so","▁som", "▁some", "▁somet","▁something",
"▁sp","▁spe", "▁st","▁start", "▁su", "▁sy", "▁t", "▁ta",
"▁take","▁talk", "▁te","▁th","▁than", "▁that", "▁the", "▁their",
"▁them", "▁then", "▁there","▁these", "▁they", "▁thing",
"▁things","▁think", "▁this","▁those","▁thousand", "▁three",
"▁through", "▁tim", "▁time", "▁to", "▁tr","▁tw", "▁two", "▁u",
"▁un","▁under", "▁up", "▁us","▁v", "▁very","▁w", "▁want", "▁was",
"▁way", "▁we", "▁well", "▁were","▁wh","▁what", "▁when",
"▁where","▁which", "▁who", "▁why", "▁will","▁with", "▁wor", "▁work",
"▁world", "▁would", "▁y","▁year","▁years", "▁you", "▁your"]

text = ["I ▁really ▁like ▁technology",
 "The ▁quick ▁brown ▁fox ▁jumps ▁over ▁the ▁lazy ▁dog.",
 "unknown chars äüößß-!$ "]
config = CtcSegmentationParameters()
config.char_list = char_list
ground_truth_mat, utt_begin_indices = prepare_text(config, text)

# ground_truth_mat
# array([[ -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190, 410,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 193, 411,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137,  13,  -1,  -1, 412,  -1,  -1,  -1,  -1,  -1],
#        [137, 140,  15,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240, 141,  -1,  16,  -1,  -1, 413,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 356,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 87,  -1, 359,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 135,  -1,  -1, 361,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [209, 438,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1, 442,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  47,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 153,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 79, 152,  -1, 154,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [188,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214, 189, 409,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 87,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  91,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  49,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 40, 266,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  -1, 275,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149, 198,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [237, 181,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1, 182,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 76, 311,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [239,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [133, 350,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [142, 220,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [183,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [204,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149, 386,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [229,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55, 230,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  69, 233,  -1, 395,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [209, 438,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83, 211, 443,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 55,  -1,  -1, 446,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [137, 356,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [241,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [240,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [244,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 52, 292,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1, 301,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 79, 152,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [214,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145, 221,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [134,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [149,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [237, 181,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [145,  -1, 182,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 43,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [ 83,  47,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [190,  24,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [204,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
#        [  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]])
```

In the example, parts of the word "▁really" were separated into the token ids: [244, 410, 411, 412, 413]
This corresponds to `['▁', '▁r', '▁re', '▁real', '▁really']`

The CTC segmentation algorithm then iterates over these tokens in the ground truth, calculates the transition probabilities for each token from `lpz` and decides for the transition(s) with the token combination that has the highest accumulated transition probability.

</div></details>


* *Sometimes the end of the last utterance is cut short. How do I solve this?* This is a known issue and strongly depends on used ASR model. A possible solution might be to just add a few milliseconds to the end of the last utterance. It's also practical to apply a threshold on the mean absolute (MA) signal, as described by [Bakhturina et al.](https://arxiv.org/abs/2104.04896).


# Reference

The full paper can be found in the preprint https://arxiv.org/abs/2007.09127 or published at <https://doi.org/10.1007/978-3-030-60276-5_27>. The code used in the paper is archived in <https://github.com/cornerfarmer/ctc_segmentation>. To cite this work:

```
@InProceedings{ctcsegmentation,
author="K{\"u}rzinger, Ludwig
and Winkelbauer, Dominik
and Li, Lujun
and Watzel, Tobias
and Rigoll, Gerhard",
editor="Karpov, Alexey
and Potapova, Rodmonga",
title="CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition",
booktitle="Speech and Computer",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="267--278",
abstract="Recent end-to-end Automatic Speech Recognition (ASR) systems demonstrated the ability to outperform conventional hybrid DNN/HMM ASR. Aside from architectural improvements in those systems, those models grew in terms of depth, parameters and model capacity. However, these models also require more training data to achieve comparable performance.",
isbn="978-3-030-60276-5"
}
```
