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


# Reference

```
@misc{ctcsegmentation,
    title={CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition},
    author={Ludwig Kürzinger and Dominik Winkelbauer and Lujun Li and Tobias Watzel and Gerhard Rigoll},
    year={2020},
    eprint={2007.09127},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
