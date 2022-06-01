# !/usr/bin/env python

"""

"""
import os
import random
from itertools import product
import numpy as np
from cgp_symbolic_regression.datasets.parent_dataset_class import ParentDataset


class DatasetPagie1(ParentDataset):
    def __init__(self):
        super().__init__()

        self.values = None
        self.labels = None

        self._make()

    def _make(self):
        self.values = product(np.arange(-5., 5.4, 0.4), np.arange(-5., 5.4, 0.4))  # evenly spaced range from [-5,
        # 5] in 0.4 intervall
        self.values = np.array(list(self.values))  # from itertools object to array
        self.labels = 1 / (1 + np.power(self.values[:, 0], -4)) + 1 / (1 + np.power(self.values[:, 1], -4))


    def get_dataset_without_split(self):
        return self.values, self.labels


if __name__ == '__main__':
    d = DatasetPagie1()
    v, l = d.get_dataset_without_split()
    print()
