#!/usr/bin/env false
# encoding: utf-8

# Copyright 2020, Technische Universität München; Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test functions for partitioning helper functions."""

from .partitioning import get_partitions


def test_get_partitions():
    """Test get_partitions.
    Performs a simple function call.
    """
    # 25 s of audio, 3s max audio length
    fs = 16000
    samples_to_frames_ratio = 4
    partitions_overlap_frames = 30
    longest_audio_segments = 3
    speech_len = 25 * fs
    partitions = get_partitions(
        speech_len,
        max_len_s=longest_audio_segments,
        samples_to_frames_ratio=samples_to_frames_ratio,
        fs=fs,
        overlap=partitions_overlap_frames,
    )
    assert partitions["cut_time_s"] == 2.4
    assert len(partitions["partitions"]) == 11
