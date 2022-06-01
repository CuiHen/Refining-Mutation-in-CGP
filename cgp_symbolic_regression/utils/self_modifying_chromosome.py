# !/usr/bin/env python

"""
"""
import random
import numpy as np
from utils.node import Node


class SelfModifyingChromosome:
    def __init__(self, params, output_type):
        self.params = params
        self.output_type = output_type

        # nodes_grid contains multiple nodes-lists. each list in nodes_grid is one column
        self.nodes_grid = list()
        # the active nodes id set contains all active nodes. However, the input nodes are not in this set
        self.active_nodes_id = set()
        self.mutations_counter = 0
        self.input_pointer = 0
        self.output_node_ids = [(self.params["graph_width"] - i, 0) for i in range(1, self.params["nbr_outputs"] + 1)]

        self._init_graph()

    def _init_graph(self):
        # fill the graph with input nodes first.
        input_nodes = list()
        for _ in range(self.params["graph_height"]):
            input_nodes.append(Node(position=0,
                                    is_input_node=True,
                                    params=self.params))
        self.nodes_grid.append(input_nodes)

        # now fill the graph with the remaining nodes
        for i in range(1, self.params["graph_width"]):
            nodes_column = list()
            for _ in range(self.params["graph_height"]):
                nodes_column.append(Node(position=i,
                                         params=self.params))

            self.nodes_grid.append(nodes_column)

        self._get_active_nodes()

    def _get_active_nodes(self):
        """
        Check, which node is active. Begin at output node and iterate from the back to the beginning
        :return:
        """
        # search for the current output nodes.
        # iterate through every node and search for output nodes. If the number of output nodes == nbr_outputs, break
        self.output_node_ids = list()
        break_flag = False
        for i in range(1, self.params["graph_width"]):
            for j in range(self.params["graph_height"]):
                if self.nodes_grid[i][j].function_id == 3:
                    # case output node
                    self.output_node_ids.append((i, j))
                if len(self.output_node_ids) >= self.params["nbr_outputs"]:
                    break_flag = True
                    break
            if break_flag:
                break
        # check if there's enough output nodes. If not, fill the rest with the last nodes.
        if len(self.output_node_ids) < self.params["nbr_outputs"]:
            remaining = self.params["nbr_outputs"] - len(self.output_node_ids)
            self.output_node_ids.extend(
                [(self.params["graph_width"] - i, 0) for i in range(1, remaining + 1)])

        # empty the active_nodes_id set and insert the output nodes
        self.active_nodes_id = set()
        for output_node in self.output_node_ids:
            self.active_nodes_id.add(output_node)

        # helper-set. while this set is not empty, there are still nodes to check.
        nodes_id_to_check = set()
        for output_node in self.output_node_ids:
            nodes_id_to_check.add(output_node)

        # start the search for every active node
        while nodes_id_to_check:
            # work from output node to input node. get the current node here
            width_nr, column_nr = nodes_id_to_check.pop()
            current_node = self.nodes_grid[width_nr][column_nr]

            if current_node.is_input_node or current_node.function_id in [0, 1, 2]:
                # case: input node. does not have predecessor
                continue

            nbr_params = current_node.get_nbr_function_params()

            # get the connection to the previous node and add it to the lists
            connection_0 = current_node.connection0_id
            nodes_id_to_check.add(connection_0)
            self.active_nodes_id.add(connection_0)

            # in case the second connection1 is used, nbr_params == 2. Then, connection1 must be considered too
            if nbr_params == 2:
                connection_1 = current_node.connection1_id
                nodes_id_to_check.add(connection_1)
                self.active_nodes_id.add(connection_1)

        # sort it for the call
        self.active_nodes_id = sorted(self.active_nodes_id)

    def __call__(self, inputs):
        # set an output grid
        # the node outputs will be saved here and used by later nodes
        outputs = {}
        for i in range(self.params["graph_width"]):
            for j in range(self.params["graph_height"]):
                outputs[(i, j)] = None

        # travers all active nodes, from input to output node
        for current_node_id in self.active_nodes_id:
            current_width, current_height = current_node_id
            current_node = self.nodes_grid[current_width][current_height]

            # case input/output node, as function IDs of 0, 1, 2 are input functions and 3 is output
            if current_node.function_id in [0, 1, 2]:
                # case INP - increment input pointer by 1 and get input
                if current_node.function_id == 0:
                    self.input_pointer = (self.input_pointer + 1) % inputs.shape[1]
                # case INPP - decrement input pointer by 1 and get input
                elif current_node.function_id == 1:
                    self.input_pointer = (self.input_pointer - 1) % inputs.shape[1]
                # case SKIPINP - increment input pointer by parameter0 and get input
                elif current_node.function_id == 2:
                    self.input_pointer = (self.input_pointer + int(current_node.parameter0)) % inputs.shape[1]

                outputs[current_node_id] = inputs[:, self.input_pointer]
            # case output node. If a node to the right is referencing an output node, treat it as an identity
            # function
            elif current_node.function_id == 3:
                outputs[current_node_id] = outputs[current_node.connection0_id]
            else:
                # case: a normal node.
                # calculate the output of the node
                connection0 = current_node.connection0_id
                connection1 = current_node.connection1_id

                outputs[current_node_id] = current_node.calc_output(outputs[connection0], outputs[connection1])

        if self.output_type == "binary":
            out = np.where(outputs[current_node_id] < 0, 1, 0)

        elif self.output_type == "scalar":
            out = outputs[current_node_id]

        else:  # categorical
            outs = list()
            for output_nodes in self.output_node_ids:
                outs.append(outputs[output_nodes])
            out = np.argmax(outs, axis=0)

        return out

    def mutate_goldman(self, nbr_active_mutations=1):
        # reset the counter
        self.mutations_counter = 0
        # mutate variables
        while True:
            # get random node id's to mutate
            node_length_id = random.randint(0, self.params["graph_width"] - 1)
            node_height_id = random.randint(0, self.params["graph_height"] - 1)

            self.nodes_grid[node_length_id][node_height_id].mutate()

            if (node_length_id, node_height_id) in self.active_nodes_id:
                self.mutations_counter += 1

                self._get_active_nodes()

                if self.mutations_counter >= nbr_active_mutations:
                    break

    def mutate_random(self, p):
        """
        Mutate every node with a probability p
        :param p:
        :return:
        """
        # Flag for the runner. If this flag is False, no active node was mutated and there's no need
        # to evaluate the chromosome a second time
        active_node_mutated = False

        for i in range(self.params["graph_width"]):
            for j in range(self.params["graph_height"]):
                if random.random() <= p:
                    self.nodes_grid[i][j].mutate()
                    # check if an active node is hit
                    if (i, j) in self.active_nodes_id:
                        active_node_mutated = True

        self._get_active_nodes()

        return active_node_mutated

    def mutate_complex_random(self, p_active, p_inactive):
        """

        :param p_active:
        :param p_inactive:
        :return:
        """
        # Flag for the runner. If this flag is False, no active node was mutated and there's no need
        # to evaluate the chromosome a second time
        active_node_mutated = False

        for i in range(self.params["graph_width"]):
            for j in range(self.params["graph_height"]):
                # case: active node:
                if (i, j) in self.active_nodes_id:
                    if random.random() <= p_active:
                        self.nodes_grid[i][j].mutate()
                        self._get_active_nodes()
                        active_node_mutated = True

                else:  # case: inactive
                    if random.random() <= p_inactive:
                        self.nodes_grid[i][j].mutate()

        return active_node_mutated
