# !/usr/bin/env python

"""

"""
import random
import numpy as np
import copy
from utils.self_modifying_chromosome import SelfModifyingChromosome
import utils.loss_metrics as custom_loss


class Island:
    def __init__(self, params, loss_type, multiclass, output_type, train_data, train_label, test_data, test_label):
        self.params = params
        self.loss_type = loss_type
        self.multiclass = multiclass
        self.mutation_type = params["mutation_type"]
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        # get the correct loss function
        if self.loss_type == "mcc":
            if self.multiclass:
                self.loss_func = custom_loss.mcc_multiclass
            else:
                self.loss_func = custom_loss.mcc
        elif self.loss_type == "summed":
            self.loss_func = custom_loss.summed_loss
        else:
            raise Exception("loss type")

        self.chromosomes = list()
        self.current_best_fitness_value = None
        self.current_fitness_values = []

        self.number_offsprings = params["nbr_offsprings"]

        for _ in range(self.number_offsprings):
            self.chromosomes.append(SelfModifyingChromosome(params, output_type))

        self.parent_id = random.randint(0, self.number_offsprings - 1)
        self.parent = self.chromosomes[self.parent_id]

        # Stuff for probablistic mutation:
        # dict to check if the an active node was changed
        self.active_node_changed = {}
        for i in range(self.number_offsprings):
            self.active_node_changed[i] = True

    def get_parent(self):
        return self.parent

    def get_current_fitness_value(self):
        return self.current_best_fitness_value

    def _calc_fitness(self, inputs, target):
        loss_dict = {}
        # iterate for every chromosome
        for i, chromosome in enumerate(self.chromosomes):
            # after the first iteration, this list should contain all fitness values of the previous iteration
            # as the parent has not changed, do not evaluate it again as the fitness value will stay the same
            if self.current_fitness_values:
                if i == self.parent_id:
                    loss_dict[i] = self.current_fitness_values[i]
                    continue

                # in case of for random mutation. If an active node changed is False, do not eval
                if not self.active_node_changed[i]:
                    loss_dict[i] = self.current_fitness_values[i]
                    continue

            # calculate the output of the current input
            outputs = chromosome(inputs)

            loss_value = self.loss_func(model_output=outputs, target=target)

            loss_dict[i] = abs(loss_value)

        return loss_dict

    def eval(self):
        chromosome_mse = self._calc_fitness(self.test_data, self.test_label)
        min_value = min(chromosome_mse.values())

        return min_value

    def evolve(self, number_goldmann_mutation=None, p=None, p_active=None, p_inactive=None):
        """
        Do one evolution step and set the new parent chromosome
        :return:
        """
        chromosome_mse = self._calc_fitness(self.train_data, self.train_label)

        # neutral search
        # get all minimum value-keys
        min_fitness_value = min(chromosome_mse.values())
        min_keys = [key for key, value in chromosome_mse.items() if value == min_fitness_value]
        # and save them to avoid unnecessary evaluations later
        self.current_best_fitness_value = min_fitness_value
        self.current_fitness_values = list(chromosome_mse.values())

        # if just one: use this one
        if len(min_keys) == 1:
            self.parent_id = min_keys[0]
        else:
            # else: remove the previous parent key and get a new one
            if self.parent_id in min_keys:
                min_keys.remove(self.parent_id)
            self.parent_id = random.choice(min_keys)

        # set the new parent
        self.parent = copy.deepcopy(self.chromosomes[self.parent_id])

        # evolve 4 new offsprings
        # the best parent stays the same
        for i in range(self.number_offsprings):
            self.chromosomes[i] = copy.deepcopy(self.parent)

            # mutate
            if i != self.parent_id:
                if self.mutation_type == "goldmann":
                    self.chromosomes[i].mutate_goldman(number_goldmann_mutation)
                elif self.mutation_type == "random":
                    self.active_node_changed[i] = self.chromosomes[i].mutate_random(p)
                elif self.mutation_type == "random_complex":
                    self.active_node_changed[i] = self.chromosomes[i].mutate_complex_random(p_active, p_inactive)
                else:
                    raise NotImplementedError("Wrong mutation type")
            else:
                self.active_node_changed[i] = False
