# !/usr/bin/env python

"""
von https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression

"""
import os
import numpy as np
import random
from cgp_classification.datasets.parent_dataset_class import ParentDataset


class DatasetHeartDisease(ParentDataset):
    def __init__(self):
        super().__init__()

        self.path = os.path.join("CGP", "cgp_classification", "datasets", "regression",
                                 "heart_disease", "framingham.csv")

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


if __name__ == '__main__':
    d = DatasetHeartDisease()
    print()
