#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020, Technische Universität München;
#                 Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Contribution: s920128 @ Github for prepare_tokenized_text()

"""CTC segmentation.

This file contains the core functions of CTC segmentation.
to extract utterance alignments within an audio file with
a given transcription.
For a description, see https://arxiv.org/abs/2007.09127
"""

import logging
import numpy as np

# import for table of character probabilities mapped to time
try:
    from .ctc_segmentation_dyn import cython_fill_table
except ImportError:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from .ctc_segmentation_dyn import cython_fill_table


class CtcSegmentationParameters:
    """Default values for CTC segmentation.

    May need adjustment according to localization or ASR settings.
    The character set is taken from the model dict, i.e., usually are generated
    with SentencePiece. An ASR model trained in the corresponding language and
    character set is needed. If the character set contains any punctuation
    characters, "#", the Greek char "ε", or the space placeholder, adapt
    these settings.
    """

    max_prob = -10000000000.0
    skip_prob = -10000000000.0
    min_window_size = 8000
    max_window_size = 100000
    subsampling_factor = 1
    frame_duration_ms = 10
    fs = None
    score_min_mean_over_L = 30
    space = "·"
    blank = 0
    replace_spaces_with_blanks = False
    blank_transition_cost_zero = False
    self_transition = "ε"
    start_of_ground_truth = "#"
    excluded_characters = ".,-?!:»«;'›‹<>()•❍·"
    tokenized_meta_symbol = "▁"
    char_list = None

    @property
    def index_duration_in_seconds(self):
        """Derive index duration from frame duration and subsampling."""
        t = self.subsampling_factor
        if self.fs is not None:
            t = t / self.fs
        else:
            t = self.frame_duration_ms * t / 1000
        return t

    @property
    def flags(self):
        """Configuration flags to pass to the table_fill operation."""
        flags = int(self.blank_transition_cost_zero)
        return flags

    def update_exluded_characters(self):
        """Remove known tokens from the list of excluded characters."""
        self.excluded_characters = "".join(
            [
                char
                for char in self.excluded_characters
                if True not in [char == j for j in self.char_list]
            ]
        )
        logging.debug(f"Excluded characters: {self.excluded_characters}")

    def __init__(self, **kwargs):
        for key in kwargs:
            if (
                not key.startswith("_")
                and hasattr(self, key)
                and kwargs[key] is not None
            ):
                setattr(self, key, kwargs[key])

    def __repr__(self):
        output = "CtcSegmentationParameters( "
        for attribute in self.__dict__.keys():
            value = self.__dict__[attribute]
            output += f"{attribute}={value}, "
        output += ")"
        return output


def ctc_segmentation(config, lpz, ground_truth):
    """Extract character-level utterance alignments.

    :param config: an instance of CtcSegmentationParameters
    :param lpz: probabilities obtained from CTC output
    :param ground_truth:  ground truth text in the form of a label sequence
    :return:
    """
    blank = 0
    audio_duration = lpz.shape[0] * config.index_duration_in_seconds
    logging.info(
        f"CTC segmentation of {len(ground_truth)} chars "
        f"to {audio_duration:.2f}s audio "
        f"({lpz.shape[0]} indices)."
    )
    if len(ground_truth) > lpz.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_size = config.min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table of alignment probabilities
        table = np.zeros(
            [min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32
        )
        table.fill(config.max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int)
        # Run actual alignment of utterances
        t, c = cython_fill_table(
            table,
            lpz.astype(np.float32),
            np.array(ground_truth),
            offsets,
            config.blank,
            config.flags,
        )
        logging.debug(
            f"Max. joint probability to align text to audio: "
            f"{table[:, c].max()} at time index {t}"
        )
        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        state_list = [""] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probs towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            if c > 0
                            else config.max_prob
                        )
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = (
                    max(lpz[t + offsets[c], blank], max_lpz_prob)
                    if t > 0
                    else config.max_prob
                )
                est_stay_prob = table[t, c] - table[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (
                                offsets[c] + t
                            ) * config.index_duration_in_seconds
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_index = ground_truth[c, min_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logging.warning(
                "IndexError: Backtracking was not successful, "
                "the window size might be too small."
            )
            window_size *= 2
            if window_size < config.max_window_size:
                logging.warning("Increasing the window size to: " + str(window_size))
                continue
            else:
                logging.error("Maximum window size reached.")
                logging.error("Check data and character list!")
                raise
        break
    return timings, char_probs, state_list


def prepare_text(config, text, char_list=None):
    """Prepare the given text for CTC segmentation.

    Creates a matrix of character symbols to represent the given text,
    then creates list of char indices depending on the models char list.

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    # temporary compatibility fix for previous espnet versions
    if type(config.blank) == str:
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace() and config.replace_spaces_with_blanks:
                if not ground_truth.endswith(config.space):
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    logging.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, blank)
            if span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    return ground_truth_mat, utt_begin_indices


def prepare_tokenized_text(config, text, char_list=None):
    """Prepare the given tokenized text for CTC segmentation.

    :param config: an instance of CtcSegmentationParameters
    :param text: string with tokens separated by spaces
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    # temporary compatibility fix for previous espnet versions
    if type(config.blank) == str:
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    ground_truth = [config.start_of_ground_truth]
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth[-1] == config.space:
            ground_truth += [config.space]
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add tokens of utterance
        for token in utt.split():
            if token in config.char_list:
                if config.replace_spaces_with_blanks and not token.beginswith(
                    config.tokenized_meta_symbol
                ):
                    ground_truth += [config.space]
                ground_truth += [token]
    # Add space to the end
    if not ground_truth[-1] == config.space:
        ground_truth += [config.space]
    logging.debug(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = 1
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] == config.space:
            ground_truth_mat[i, 0] = config.blank
        else:
            char_index = config.char_list.index(ground_truth[i])
            ground_truth_mat[i, 0] = char_index
    return ground_truth_mat, utt_begin_indices


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text):
    """Utterance-wise alignments from char-wise alignments.

    :param config: an instance of CtcSegmentationParameters
    :param utt_begin_indices: list of time indices of utterance start
    :param char_probs:  character positioned probabilities obtained from backtracking
    :param timings: mapping of time indices to seconds
    :param text: list of utterances
    :return: segments, a list of: utterance start and end [s], and its confidence score
    """

    def compute_time(index, align_type):
        """Compute start and end time of utterance.

        :param index:  frame index value
        :param align_type:  one of ["begin", "end"]
        :return: start/end time of utterance in seconds
        """
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)

    segments = []
    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start / config.index_duration_in_seconds))
        end_t = int(round(end / config.index_duration_in_seconds))
        # Compute confidence score by using the min mean probability
        #   after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t == start_t:
            min_avg = 0
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = 0
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments
