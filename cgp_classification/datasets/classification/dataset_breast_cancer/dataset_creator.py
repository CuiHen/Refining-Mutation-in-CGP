# !/usr/bin/env python

"""
https://archive-beta.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+diagnostic
"""
import os
import random

import numpy as np

from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetBreastCancer(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "classification",
                                 "dataset_breast_cancer", "breast-cancer-wisconsin.data")

        self._make()

    def _make(self, scaling=None):
        with open(self.path, "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(",") for l in lines]  # split them into their respective values
        lines = [l[1:] for l in lines]  # remove the ID number as it is not an attribute for training
        random.shuffle(lines)  # and shuffle

        self._split_train_test(lines, scaling=scaling)

        # 2 is benign, 4 is malignant
        # map to 0 for benign, 1 for malignant
        self.label_train = np.where(self.label_train == 2, 0, 1)
        self.label_test = np.where(self.label_test == 2, 0, 1)


if __name__ == '__main__':
    d = DatasetBreastCancer()
    print(np.shape(d.get_test()[1]))
