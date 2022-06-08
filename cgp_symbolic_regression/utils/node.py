# !/usr/bin/env python

"""
"""
import random

from utils.self_modifying_functions import SelfModifyingFunctionList


class Node:
    def __init__(self, params, position, is_input_node=False, levels_back=-1):
        self.position = position
        self.is_input_node = is_input_node
        self.params = params
        self.param0_range = params["parameter_0_range"]

        if levels_back == -1:
            self.levels_back = position - 1
        else:
            self.levels_back = levels_back

        self.functions_list = SelfModifyingFunctionList()

        self.function_id = -1
        self.connection0_id = -1
        self.connection1_id = -1
        self.parameter0 = -1

        self._get_random_function_id()
        self._get_random_param_0()

        if not is_input_node:
            self._get_random_connection_0_id()
            self._get_random_connection_1_id()

    def _get_random_function_id(self):
        if self.is_input_node:
            self.function_id = self._generate_random_number(0, 2, self.function_id)
        else:
            self.function_id = self._generate_random_number(0, len(self.functions_list) - 1, self.function_id)

    def _get_random_connection_0_id(self):
        if not self.is_input_node:
            random_pos = self._generate_random_number((self.position - 1) - self.levels_back,
                                                      self.position - 1,
                                                      self.connection0_id)
            width = max(0, random_pos)
            self.connection0_id = (width,
                                   random.randint(0, self.params["graph_height"] - 1))

    def _get_random_connection_1_id(self):
        if not self.is_input_node:
            random_pos = self._generate_random_number((self.position - 1) - self.levels_back,
                                                      self.position - 1,
                                                      self.connection1_id)
            width = max(0, random_pos)
            self.connection1_id = (width,
                                   random.randint(0, self.params["graph_height"] - 1))

    def _get_random_param_0(self):
        self.parameter0 = random.uniform(self.param0_range[0], self.param0_range[1])

    def _get_function(self):
        return self.functions_list.function_dict[self.function_id]

    def get_nbr_function_params(self):
        return self.functions_list.nbr_params_dict[self.function_id]

    def calc_output(self, connection0_value, connection1_value):
        res = self._get_function()(connection0_value,
                                   connection1_value,
                                   self.parameter0)
        return res

    def mutate(self):
        param = random.randint(0, 3)

        if param == 0:
            self._get_random_function_id()
        elif param == 1:
            self._get_random_connection_0_id()
        elif param == 2:
            self._get_random_connection_1_id()
        elif param == 3:
            # Veränderung param0
            # Entweder ganz neu, oder um max +-10% verändern
            if bool(random.getrandbits(1)):
                self._get_random_param_0()
            else:
                # erstelle eine zahl zwischen -0.1 und +0.1, um jeweils max 10 prozent dazu addieren oder subtrahieren
                # zu können
                percent = random.uniform(-.1, .1)
                self.parameter0 = self.parameter0 * percent + self.parameter0

    def _generate_random_number(self, lower_limit, upper_limit, excluded):
        """

        :param lower_limit:
        :param upper_limit:
        :param excluded:
        :return:
        """
        while True:
            random_number = random.randint(lower_limit, upper_limit)
            if random_number != excluded or excluded <= 1:
                break

        return random_number


class InputNode(Node):
    def __init__(self, params, position, input_id):
        super().__init__(params, position, is_input_node=True)
        self.input_id = input_id
