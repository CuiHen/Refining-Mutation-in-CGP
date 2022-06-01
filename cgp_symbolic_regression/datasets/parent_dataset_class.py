# !/usr/bin/env python

"""

"""
import os
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


class ParentDataset:
    def __init__(self):
        self.attributes_train = []
        self.label_train = []
        self.attributes_test = []
        self.label_test = []

    def get_train(self):
        return self.attributes_train, self.label_train

    def get_test(self):
        return self.attributes_test, self.label_test

    def get_dataset_without_split(self):
        return np.vstack((self.attributes_train, self.attributes_test)), np.hstack((self.label_train, self.label_test))

