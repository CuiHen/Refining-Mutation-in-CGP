# !/usr/bin/env python

"""
"""
import os
from time import time
from multiprocessing import Process
import sys

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from runner.island_runner import Island
from utils.params import params

from datasets.classification.dataset_breast_cancer.dataset_creator import DatasetBreastCancer
from datasets.classification.abalone.dataset_creator import DatasetAbalone
from datasets.classification.page_block.dataset_creator import DatasetPageBlock
from datasets.classification.credit.dataset_creator import DatasetCredit
from datasets.classification.spect.dataset_creator import DatasetSpect
from datasets.regression.forest_fire.dataset_creator import DatasetFire
from datasets.regression.heart_disease.dataset_creator import DatasetHeartDisease


def learn(current_run_nbr, train_attributes, test_attributes, train_labels, test_labels, loss_type,
          multiclass_classification, output_type, path, params):
    n_repeat = current_run_nbr // 5
    n_split = current_run_nbr % 5

    # open a text file to log everything
    handle = open(os.path.join(path, "n_repeat_{}_n_split_{}.txt".format(n_repeat, n_split)), "w")
    handle.write("{}\n".format(params))

    # init new islands
    cgp_runner = Island(params, loss_type, multiclass_classification, output_type, train_attributes, train_labels,
                        test_attributes, test_labels)

    # start the main training loop:
    best_test_fitness = 1e10
    for i in range(params["iterations"]):
        # start the evolution steps and the processes for evolution
        # cgp_runner.evolve(p_active=params["p_active"], p_inactive=params["p_inactive"])
        cgp_runner.evolve(number_goldmann_mutation=params["number_goldmann_mutation"])

        # print me something spicy
        if i % 50 == 0:
            # get best fitness
            fitness = cgp_runner.get_current_fitness_value()

            handle.write("TRAIN: Iteration: {}, min mcc value: {}\n".format(i, fitness))

        # evaluation step with the best chromosome
        if i % params["eval_after_iterations"] == 0:
            # get best island
            fitness = cgp_runner.eval()

            handle.write("TEST: Iteration: {}, min mcc value: {}\n".format(i, fitness))

            if fitness < best_test_fitness:
                best_test_fitness = fitness

            if fitness < 0.01:
                break

    handle.write("\n")
    handle.write("#" * 20)
    handle.write("\n")
    handle.write("Iteration: {}\n".format(i))
    handle.write("Best fitness\n")
    handle.write("{}".format(best_test_fitness))

    handle.close()

    return


def main():
    params["number_goldmann_mutation"] = None
    params["p"] = None
    params["p_active"] = None
    params["p_inactive"] = None

    dataset_nbr = int(sys.argv[1])
    # classification
    if dataset_nbr == 0:
        params["dataset"] = "breast_cancer"
        dataset = DatasetBreastCancer()
        multiclass_classification = False
        loss_type = "mcc"
        output_type = "binary"
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif dataset_nbr == 1:
        params["dataset"] = "page_block"
        dataset = DatasetPageBlock()
        multiclass_classification = True
        loss_type = "mcc"
        output_type = "categorical"
        params["nbr_outputs"] = 5
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif dataset_nbr == 2:
        params["dataset"] = "abalone"
        dataset = DatasetAbalone()
        multiclass_classification = False
        loss_type = "summed"
        output_type = "scalar"
        rskf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif dataset_nbr == 3:
        params["dataset"] = "credit"
        dataset = DatasetCredit()
        multiclass_classification = False
        loss_type = "mcc"
        output_type = "binary"
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif dataset_nbr == 4:
        params["dataset"] = "spect"
        dataset = DatasetSpect()
        multiclass_classification = False
        loss_type = "mcc"
        output_type = "binary"
        rskf = None
    # regression
    elif dataset_nbr == 5:
        params["dataset"] = "forest_fire"
        dataset = DatasetFire()
        multiclass_classification = None
        loss_type = "summed"
        output_type = "scalar"
        rskf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif dataset_nbr == 6:
        params["dataset"] = "heart_disease"
        dataset = DatasetHeartDisease()
        multiclass_classification = False
        loss_type = "mcc"
        output_type = "binary"
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    else:
        raise NotImplementedError("Dataset , etc. slurm nbr. {}".format(dataset_nbr))

    # params["p_active"] = float(sys.argv[2])
    # params["p_inactive"] = float(sys.argv[3])
    params["number_goldmann_mutation"] = int(sys.argv[2])

    # make path
    task = "prob_complex_active_{}_inactive_{}".format(params["p_active"], params["p_inactive"])
    path = "./outs/{}/{}_vanilla".format(task, params["dataset"])
    os.makedirs(path, exist_ok=True)

    print(params)

    ###############################################################################
    # start the training with kfold or without
    ###############################################################################
    if rskf is not None:
        attributes, labels = dataset.get_dataset_without_split()

        params["nbr_inputs"] = len(attributes[0])

        working_processes = list()
        for current_run_nbr, (train_index, test_index) in enumerate(rskf.split(attributes, labels)):
            attributes_train, attributes_test = attributes[train_index], attributes[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            p = Process(target=learn, args=(current_run_nbr,
                                            attributes_train,
                                            attributes_test,
                                            labels_train,
                                            labels_test,
                                            loss_type,
                                            multiclass_classification,
                                            output_type,
                                            path,
                                            params))
            p.start()
            working_processes.append(p)

        for i in range(len(working_processes)):
            working_processes[i].join()

    elif rskf is None:
        attributes_train, labels_train = dataset.get_train()
        attributes_test, labels_test = dataset.get_test()

        params["nbr_inputs"] = len(attributes_train[0])

        working_processes = list()
        for current_run_nbr in range(5):
            p = Process(target=learn, args=(current_run_nbr,
                                            attributes_train,
                                            attributes_test,
                                            labels_train,
                                            labels_test,
                                            loss_type,
                                            multiclass_classification,
                                            output_type,
                                            path,
                                            params))
            p.start()
            working_processes.append(p)

        for i in range(len(working_processes)):
            working_processes[i].join()

    print("DONE")


if __name__ == '__main__':
    print(params)

    main()
