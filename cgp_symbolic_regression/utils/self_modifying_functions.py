# !/usr/bin/env python

"""

"""
import sys

import cv2
import numpy as np
import math
import warnings

warnings.filterwarnings("error")


class SelfModifyingFunctionList:
    def __init__(self):
        super().__init__()
        self.function_dict = {
            0: "INP",
            1: "INPP",
            2: "SKIP",
            3: "OUTPUT",
            4: self.add,
            5: self.sub,
            6: self.mul,
            7: self.div,
            8: self.sin,
            9: self.cos,
            10: self.log,
            11: self.exp,
        }

        self.nbr_params_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
        }

    def __len__(self):
        return len(self.function_dict)

    def add(self, connection0, connection1, param0):
        return connection0 + connection1

    def sub(self, connection0, connection1, param0):
        return connection0 - connection1

    def mul(self, connection0, connection1, param0):
        return connection0 * connection1

    def div(self, connection0, connection1, param0):
        return np.divide(connection0, connection1, out=np.zeros_like(connection0), where=(connection1 != 0))

    def sin(self, connection0, connection1, param0):
        return np.sin(connection0)

    def cos(self, connection0, connection1, param0):
        return np.cos(connection0)

    def log(self, connection0, connection1, param0):
        return np.log(np.abs(connection0), out=np.zeros_like(connection0), where=(connection0 != 0))

    def exp(self, connection0, connection1, param0):
        return np.exp(connection0)


if __name__ == '__main__':
    f = SelfModifyingFunctionList()
    import numpy as np

    a = np.arange(256).astype(np.uint8)
    for i in range(len(f)):
        res = f.function_dict[i](a, a, 1, 1)
        print(res.dtype)
