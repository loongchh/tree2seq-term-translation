# -*- encoding: utf-8 -*-
from collections import Counter
import string
import re
import numpy as np
import argparse
import os
import time
import shutil
import sys
import codecs

from onmt.utils.logging import init_logger, logger

translator = str.maketrans('', '', string.punctuation)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        # exclude = set(string.punctuation)
        # return ''.join(ch for ch in text if ch not in exclude)
        return text.translate(translator)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def score_line(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (0, 0, 0)
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


def test_f1(cand, ref):
    """Calculate F1 scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    scores = [score_line(lp, lt) for lp, lt in zip(cand, ref)]
    em = [1 if s[0] == 1 else 0 for s in scores]

    results = {}
    f1, precision, recall = zip(*scores)
    results['f1'] = np.array(f1).mean()
    results['precision'] = np.array(precision).mean()
    results['recall'] = np.array(recall).mean()
    results['em'] = np.array(em).mean()
    return results


def results_to_str(results):
    return ">> F1/Precision/Recall: {:.2f}/{:.2f}/{:.2f}\n>> EM: {:.2f}".format(
        results["f1"] * 100,
        results["precision"] * 100,
        results["recall"] * 100,
        results["em"] * 100)


if __name__ == "__main__":
    init_logger('test_f1.log')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', type=str, default="candidate.txt", help='candidate file')
    parser.add_argument(
        '-r', type=str, default="reference.txt", help='reference file')
    args = parser.parse_args()
    if args.c.upper() == "STDIN":
        candidates = sys.stdin
    else:
        candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    results = test_f1(candidates, references)
    print(results_to_str(results))
