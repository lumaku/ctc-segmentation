#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020, Technische Universität München; Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Audio Partitioning Tool for CTC Segmentation.

Sometimes, The inference of the model is too slow and uses too much memory.
Use RNN-based ASR networks that grow linearly in complexity with longer
audio files. Inference complexity increases on long audio files quadratically
for Transformer-based architectures. To solve this, it is possible to partition
the audio into several parts. This file includes a helper function.
"""


def get_partitions(
    t: int = 100000,
    max_len_s: float = 1280.0,
    fs: int = 16000,
    samples_to_frames_ratio=512,
    overlap: int = 0,
):
    """Obtain partitions.

    Note that this is implemented for frontends that discard trailing data.
    Note that the partitioning strongly depends on your architecture.
    A note on audio indices:
        Based on the ratio of audio sample points to lpz indices (here called
        frame), the start index of block N is:
        0 + N * samples_to_frames_ratio
        Due to the discarded trailing data, the end is then in the range of:
        [N * samples_to_frames_ratio - 1 .. (1+N) * samples_to_frames_ratio]
    """
    # max length should be ~ cut length + 25%
    cut_time_s = max_len_s / 1.25
    max_length = int(max_len_s * fs)
    cut_length = int(cut_time_s * fs)
    # make sure its a multiple of frame size
    max_length -= max_length % samples_to_frames_ratio
    cut_length -= cut_length % samples_to_frames_ratio
    overlap = int(max(0, overlap))
    if (max_length - cut_length) <= samples_to_frames_ratio * (2 + overlap):
        raise ValueError(
            f"Pick a larger time value for partitions. "
            f"time value: {max_len_s}, "
            f"overlap: {overlap}, "
            f"ratio: {samples_to_frames_ratio}."
        )
    partitions = []
    duplicate_frames = []
    cumulative_lpz_length = 0
    cut_length_lpz_frames = int(cut_length // samples_to_frames_ratio)
    partition_start = 0
    while t > max_length:
        start = partition_start - samples_to_frames_ratio * overlap
        start = int(max(0, start))
        end = cut_length + samples_to_frames_ratio * (1 + overlap) - 1
        end = int(partition_start + end)
        partitions += [(start, end)]
        # overlap - duplicate frames shall be deleted.
        cumulative_lpz_length += cut_length_lpz_frames
        for i in range(overlap):
            duplicate_frames += [
                cumulative_lpz_length - i,
                cumulative_lpz_length + (1 + i),
            ]
        # next partition
        t -= cut_length
        partition_start += cut_length
    else:
        start = partition_start - samples_to_frames_ratio * overlap
        start = int(max(0, start))
        partitions += [(start, None)]
    partition_dict = {
        "partitions": partitions,
        "overlap": overlap,
        "delete_overlap_list": duplicate_frames,
        "samples_to_frames_ratio": samples_to_frames_ratio,
        "max_length": max_length,
        "cut_length": cut_length,
        "cut_time_s": cut_time_s,
    }
    return partition_dict
