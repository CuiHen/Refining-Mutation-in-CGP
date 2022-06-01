# !/usr/bin/env python

"""

"""
import os
import numpy as np
import random
from cgp_symbolic_regression.datasets.parent_dataset_class import ParentDataset


class Vladislavleva4(ParentDataset):
    def __init__(self):
        super().__init__()

        self.attributes_train = None
        self.label_train = None

        self.attributes_test = None
        self.label_test = None

        self._make()

    def _make(self):
        self.attributes_train = np.random.uniform(0.05, 6.05, (1024, 5))
        self.attributes_test = np.random.uniform(-0.25, 6.35, (50_000, 5))
        self.label_train = 10 / (5 + np.sum(np.square(self.attributes_train - 3), axis=1))
        self.label_test = 10 / (5 + np.sum(np.square(self.attributes_test - 3), axis=1))

    def get_dataset_without_split(self):
        return None


if __name__ == '__main__':
    d = Vladislavleva4()
    print()
