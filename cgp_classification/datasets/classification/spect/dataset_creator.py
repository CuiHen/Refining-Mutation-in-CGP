# !/usr/bin/env python

"""
https://archive.ics.uci.edu/ml/datasets/spect+heart
"""
import os
import random

import numpy as np

from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetSpect(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "classification",
                                 "spect")

        self._make_train()
        self._make_test()

    def _make_train(self):
        with open(os.path.join(self.path, "SPECT.train"), "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        random.shuffle(lines)  # and shuffle

        converted_lines = list()
        for line in lines:
            converted_lines.append(list(map(float, line)))

        lines = np.array(converted_lines)
        attributes = lines[:, 1:]
        labels = lines[:, 0]

        attributes = self._standardize_input(attributes)
        self.attributes_train = attributes
        self.label_train = labels

    def _make_test(self):
        with open(os.path.join(self.path, "SPECT.test"), "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        random.shuffle(lines)  # and shuffle

        converted_lines = list()
        for line in lines:
            converted_lines.append(list(map(float, line)))

        lines = np.array(converted_lines)
        attributes = lines[:, 1:]
        labels = lines[:, 0]

        attributes = self._standardize_input(attributes)
        self.attributes_test = attributes
        self.label_test = labels

    def get_dataset_without_split(self):
        return None


if __name__ == '__main__':
    d = DatasetSpect()
    print(np.unique(d.get_test()[1]))
    print(np.unique(d.get_train()[1]))
    print(np.shape(d.get_train()[0]))
    print(np.shape(d.get_test()[0]))
