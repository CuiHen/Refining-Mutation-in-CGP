# !/usr/bin/env python

"""
imports are not pretty

"""
import os
from multiprocessing import Process
import sys

import numpy as np
from sklearn.model_selection import RepeatedKFold
from runner.island_runner import Island
from utils.params import params

from datasets.symbolic_regression import nguyen7, vladislavleva4, pagie1

def learn(current_run_nbr, train_attributes, test_attributes, train_labels, test_labels, loss_type,
          multiclass_classification, output_type, path, reduction, params):
    n_repeat = current_run_nbr // 5
    n_split = current_run_nbr % 5

    # open a text file to log everything
    handle = open(os.path.join(path, "n_repeat_{}_n_split_{}.txt".format(n_repeat, n_split)), "w")
    handle.write("{}\n".format(params))

    # init new islands
    cgp_runner = Island(params, loss_type, multiclass_classification, output_type, train_attributes, train_labels,
                        test_attributes, test_labels)

    # start the main training loop:
    decrease_every_n_steps = params["iterations"] // params["nbr_goldman_mutations"]
    best_test_fitness = 1e10
    for i in range(params["iterations"]):
        # start the evolution steps and the processes for evolution
        cgp_runner.evolve(number_goldmann_mutation=params["nbr_goldman_mutations"])

        # decrease the number of active nodes to hit for the goldman mutation
        if reduction:
            if i != 0 and i % decrease_every_n_steps == 0:
                params["nbr_goldman_mutations"] -= 1

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
    # np.random.seed(0)
    # random.seed(0)
    # cv2.setRNGSeed(0)
    params["number_goldmann_mutation"] = None
    params["p"] = None
    params["p_active"] = None
    params["p_inactive"] = None

    slurm_nbr = int(sys.argv[1])
    if slurm_nbr == 0:
        params["dataset"] = "nguyen7"
        dataset = nguyen7.DatasetNguyen7()
        multiclass_classification = False
        loss_type = "summed"
        output_type = "scalar"
        rskf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
        # rskf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif slurm_nbr == 1:
        params["dataset"] = "vlad4"
        dataset = vladislavleva4.Vladislavleva4()
        multiclass_classification = False
        loss_type = "summed"
        output_type = "scalar"
        rskf = None
    elif slurm_nbr == 2:
        params["dataset"] = "pagie1"
        dataset = pagie1.DatasetPagie1()
        multiclass_classification = False
        loss_type = "summed"
        output_type = "scalar"
        rskf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
        # rskf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    else:
        raise NameError("Wrong dataset")

    nbr_goldman_mutations = int(sys.argv[2])
    params["nbr_goldman_mutations"] = nbr_goldman_mutations
    reduction = bool(int(sys.argv[3]))
    params["reduction"] = reduction

    print(params)

    # make path
    if reduction:
        task = "goldmann_mutation_{}_with_decrease".format(nbr_goldman_mutations)
    else:
        task = "goldmann_mutation_{}_no_decrease".format(nbr_goldman_mutations)
    print(task)
    path = "./BIOMA/{}/{}_vanilla".format(task, params["dataset"])
    os.makedirs(path, exist_ok=True)

    print(params)

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
                                            reduction,
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
                                            reduction,
                                            params))
            p.start()
            working_processes.append(p)

        for i in range(len(working_processes)):
            working_processes[i].join()

    print("DONE")


if __name__ == '__main__':
    print(params)

    main()
