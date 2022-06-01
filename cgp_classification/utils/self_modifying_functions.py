# !/usr/bin/env python

"""

"""


class SelfModifyingFunctionList:
    def __init__(self):
        super().__init__()
        self.function_dict = {
            0: "INP",
            1: "INPP",
            2: "SKIP",
            3: "OUTPUT",
            4: self.add,
            5: self.add_c,
            6: self.sub,
            7: self.sub_c,
            8: self.mul,
            9: self.mul_c,
            10: self.div,
            11: self.div_c,
            12: self.sin,
            13: self.cos,
            14: self.tan,
            15: self.tanh,
            16: self.log,
            17: self.log1p,
            18: self.sqrt,
            19: self.abs,
            20: self.ceil,
            21: self.floor,
            22: self.max,
            23: self.min,
            24: self.greater,
            25: self.greater_c,
            26: self.lesser,
            27: self.lesser_c,
            28: self.negate,
        }

        self.nbr_params_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 2,
            5: 1,
            6: 2,
            7: 1,
            8: 2,
            9: 1,
            10: 2,
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 1,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            24: 2,
            25: 1,
            26: 2,
            27: 1,
            28: 1,
            29: 1,
        }

    def __len__(self):
        return len(self.function_dict)

    def add(self, connection0, connection1, param0):
        return connection0 + connection1

    def add_c(self, connection0, connection1, param0):
        return connection0 + param0

    def sub(self, connection0, connection1, param0):
        return connection0 - connection1

    def sub_c(self, connection0, connection1, param0):
        return connection0 - param0

    def mul(self, connection0, connection1, param0):
        return connection0 * connection1

    def mul_c(self, connection0, connection1, param0):
        return connection0 * param0

    def div(self, connection0, connection1, param0):
        return np.divide(connection0, connection1, out=np.zeros_like(connection0), where=(connection1 != 0))

    def div_c(self, connection0, connection1, param0):
        if param0 == 0:
            return np.zeros_like(connection0)
        return connection0 / param0

    def sin(self, connection0, connection1, param0):
        return np.sin(connection0)

    def cos(self, connection0, connection1, param0):
        return np.cos(connection0)

    def tan(self, connection0, connection1, param0):
        return np.tan(connection0)

    def tanh(self, connection0, connection1, param0):
        return np.tanh(connection0)

    def log(self, connection0, connection1, param0):
        return np.log(np.abs(connection0), out=np.zeros_like(connection0), where=(connection0 != 0))

    def log1p(self, connection0, connection1, param0):
        return np.log1p(np.abs(connection0), out=np.zeros_like(connection0), where=(connection0 != 0))

    def sqrt(self, connection0, connection1, param0):
        return np.sqrt(abs(connection0))

    def abs(self, connection0, connection1, param0):
        return np.fabs(connection0)

    def ceil(self, connection0, connection1, param0):
        return np.ceil(connection0)

    def floor(self, connection0, connection1, param0):
        return np.floor(connection0)

    def max(self, connection0, connection1, param0):
        return np.maximum(connection0, np.full(connection0.shape, param0, dtype=np.float32))

    def min(self, connection0, connection1, param0):
        return np.minimum(connection0, np.full(connection0.shape, param0, dtype=np.float32))

    def greater(self, connection0, connection1, param0):
        return np.greater(connection0, connection1).astype(np.float32)

    def greater_c(self, connection0, connection1, param0):
        return np.greater(connection0, np.full(connection0.shape, param0, dtype=np.float32)).astype(np.float32)

    def lesser(self, connection0, connection1, param0):
        return np.less(connection0, connection1).astype(np.float32)

    def lesser_c(self, connection0, connection1, param0):
        return np.less(connection0, np.full(connection0.shape, param0, dtype=np.float32)).astype(np.float32)

    def const(self, connection0, connection1, param0):
        return param0

    def negate(self, connection0, connection1, param0):
        return -connection0


if __name__ == '__main__':
    f = SelfModifyingFunctionList()
    import numpy as np

    a = np.arange(256).astype(np.uint8)
    for i in range(len(f)):
        res = f.function_dict[i](a, a, 1, 1)
        print(res.dtype)
