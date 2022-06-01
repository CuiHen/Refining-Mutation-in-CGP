# !/usr/bin/env python

"""
https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
"""
import os
import random

import numpy as np

from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetPageBlock(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "classification",
                                 "page_block", "page-blocks.data")

        self._make()

    def _make(self, scaling="standardize"):
        with open(self.path, "r") as handle:
            lines = handle.readlines()

        lines = [l.replace("\n", "") for l in lines]  # remove linebreak
        lines = [l.split(" ") for l in lines]  # split them into their respective values
        # as there're a lot of empty strings due to the seperator being x-many spaces, remove them
        lines = [list(filter(None, l)) for l in lines]
        random.shuffle(lines)  # and shuffle

        self._split_train_test(lines, scaling=scaling)

        # as the labels are in range [1, 5], map them to [0, 4]
        self.label_test -= 1
        self.label_train -= 1


if __name__ == '__main__':
    d = DatasetPageBlock()
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    attributes, labels = d.get_dataset_without_split()
    for train_index, test_index in rskf.split(attributes, labels):
        print(np.unique(labels[test_index]), np.unique(labels[train_index]))
