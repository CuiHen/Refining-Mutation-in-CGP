# !/usr/bin/env python

"""

"""
import os
import numpy as np
import random
from cgp_symbolic_regression.datasets.parent_dataset_class import ParentDataset


class DatasetNguyen7(ParentDataset):
    def __init__(self):
        super().__init__()

        self.values = None
        self.labels = None

        self._make()

    def _make(self):
        self.values = np.random.uniform(0, 2, (20, 1))
        self.labels = np.log(self.values + 1) + np.log(np.square(self.values) + 1)
        self.labels = np.squeeze(self.labels)

    def get_dataset_without_split(self):
        return self.values, self.labels


if __name__ == '__main__':
    d = DatasetNguyen7()
    v, l = d.get_dataset_without_split()
    print()
