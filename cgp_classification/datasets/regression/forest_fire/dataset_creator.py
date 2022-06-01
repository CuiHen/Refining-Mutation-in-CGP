# !/usr/bin/env python

"""
https://archive.ics.uci.edu/ml/datasets/forest+fires
"""
import os
import numpy as np
import random
from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetFire(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "regression",
                                 "forest_fire", "forestfires.csv")
        self._make()

    def _make(self, scaling="standardize"):
        with open(self.path, "r") as handle:
            lines = handle.readlines()

        # remove the attribute description of the csv file
        lines = lines[1:]
        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        # convert month to int
        random.shuffle(lines)  # shuffle as it is sorted by glass type

        self._split_train_test(lines, scaling=scaling)

        self.label_train = np.log(self.label_train + 1)
        self.label_test = np.log(self.label_test + 1)


if __name__ == '__main__':
    d = DatasetFire()
    print()
