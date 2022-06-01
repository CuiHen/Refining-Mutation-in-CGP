# !/usr/bin/env python

"""
https://archive.ics.uci.edu/ml/datasets/credit+approval
"""
import os
import random

import numpy as np

from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetCredit(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "classification",
                                 "credit", "crx.data")

        self._make()

    def _make(self, scaling="standardize"):
        with open(self.path, "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        random.shuffle(lines)  # and shuffle

        self._split_train_test(lines, scaling=scaling)


if __name__ == '__main__':
    d = DatasetCredit()
    print(np.unique(d.get_test()[1]))
    print(np.unique(d.get_train()[1]))
