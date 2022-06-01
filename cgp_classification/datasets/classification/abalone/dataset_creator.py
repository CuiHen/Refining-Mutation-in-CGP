# !/usr/bin/env python

"""
https://archive.ics.uci.edu/ml/datasets/Abalone
"""
import os
import random

import numpy as np

from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetAbalone(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "classification",
                                 "abalone", "abalone.data")

        self._make()

    def _make(self, scaling="standardize"):
        with open(self.path, "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        random.shuffle(lines)  # and shuffle

        self._split_train_test(lines, scaling=scaling)

        # as number rings +1.5 gives the age in years
        self.label_test += 1.5
        self.label_train += 1.5


if __name__ == '__main__':
    d = DatasetAbalone()
    print(np.shape(d.get_test()[1]))
