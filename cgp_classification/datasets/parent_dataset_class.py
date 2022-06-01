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

    def _normalize_input(self, inputs):
        # transpose so every row corresponds to one attribute
        inputs = inputs.T
        for i in range(len(inputs)):
            inputs[i] = (inputs[i] - np.min(inputs[i])) / np.ptp(inputs[i])

        # transform back
        inputs = inputs.T

        return inputs

    def _standardize_input(self, inputs):
        # transpose so every row corresponds to one attribute
        inputs = inputs.T
        for i in range(len(inputs)):
            inputs[i] = (inputs[i] - np.mean(inputs[i])) / np.std(inputs[i])

        # transform back
        inputs = inputs.T

        return inputs

    def _split_train_test(self, lines):
        converted_lines = list()
        for line in lines:
            if "?" in line:
                continue
            converted_lines.append(list(map(float, line)))

        lines = np.array(converted_lines)
        attributes = lines[:, :-1]
        labels = lines[:, -1]

        attributes = self._standardize_input(attributes)

        # split about half for train / test
        split_at = len(attributes) // 2
        self.attributes_train = attributes[:split_at]
        self.attributes_test = attributes[split_at:]
        self.label_train = labels[:split_at]
        self.label_test = labels[split_at:]
