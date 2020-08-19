# CTC segmentation


CTC segmentation can be used to find utterances alignments within large audio files.

* This repository contains the `ctc-segmentation` python package.
* The complete code is in https://github.com/cornerfarmer/ctc_segmentation
* A description of the algorithm is in https://arxiv.org/abs/2007.09127

# Example Code

1. `prepare_text` filters characters not in the dictionary, and generates the character matrix.
2. `ctc_segmentation` computes character-wise alignments from CTC activations of an already trained CTC-based network.
3. `determine_utterance_segments` converts char-wise alignments to utterance-wise alignments.
4. In a post-processing step, segments may be filtered by their confidence value.

This code is from `asr_align.py` of the ESPnet toolkit:


```python
from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text

...


config = CtcSegmentationParameters()
char_list = train_args.char_list
config.blank = char_list[0]

for idx, name in enumerate(js.keys(), 1):
    logging.info("(%d/%d) Aligning " + name, idx, len(js.keys()))
    batch = [(name, js[name])]
    feat, label = load_inputs_and_targets(batch)
    feat = feat[0]
    with torch.no_grad():
        # Encode input frames
        enc_output = model.encode(torch.as_tensor(feat).to(device)).unsqueeze(0)
        # Apply ctc layer to obtain log character probabilities
        lpz = model.ctc.log_softmax(enc_output)[0].cpu().numpy()
    # Prepare the text for aligning
    ground_truth_mat, utt_begin_indices = prepare_text(
        config, text[name], char_list
    )
    # Align using CTC segmentation
    timings, char_probs, state_list = ctc_segmentation(
        config, lpz, ground_truth_mat
    )
    # Obtain list of utterances with time intervals and confidence score
    segments = determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, text[name]
    )
    # Write to "segments" file
    for i, boundary in enumerate(segments):
        utt_segment = (
            f"{segment_names[name][i]} {name} {boundary[0]:.2f}"
            f" {boundary[1]:.2f} {boundary[2]:.9f}\n"
        )
        args.output.write(utt_segment)
```

After the segments are written to a `segments` file, they can be filtered with the parameter `min_confidence_score`. This is minium confidence score in log space as described in the paper. Utterances with a low confidence score are discarded. This parameter may need adjustment depending on dataset, ASR model and language. For the german ASR model, a value of -1.5 worked very well, but for TEDlium, a lower value of about -5.0 seemed more practical.

```bash
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${unfiltered} > ${filtered}
```

# Parameters

There are several notable parameters to adjust the working of the algorithm:


* `min_window_size`: Minimum window size considered for a single utterance. The current default value should be OK in most cases.

* Localization: The character set is taken from the model dict, i.e., usually are generated with SentencePiece. An ASR model trained in the corresponding language and character set is needed. For asian languages, no changes to the CTC segmentation parameters should be necessary. One exception: If the character set contains any punctuation characters, "#", or the Greek char "ε", adapt the setting in an instance of `CtcSegmentationParameters` in `segmentation.py`.

* `CtcSegmentationParameters` includes a blank character. Copy over the Blank character from the dictionary to the configuration, if in the model dictionary e.g. "\<blank>" instead of the default "_" is used. If the Blank in the configuration and in the dictionary mismatch, the algorithm raises an IndexError at backtracking.

Two parameters are needed to correctly map the frame indices to a time stamp in seconds:

* `subsampling_factor`: If the encoder sub-samples its input, the number of frames at the CTC layer is reduced by this factor. A BLSTMP encoder with subsampling 1_2_2_1_1 has a subsampling factor of 4. 
* `frame_duration`: This is the non-overlapping duration of a single frame in milliseconds (the inverse of frames per millisecond).


# Reference

```
@misc{ctcsegmentation,
    title={CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition},
    author={Ludwig K{\u}rzinger and Dominik Winkelbauer and Lujun Li and Tobias Watzel and Gerhard Rigoll},
    year={2020},
    eprint={2007.09127},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
