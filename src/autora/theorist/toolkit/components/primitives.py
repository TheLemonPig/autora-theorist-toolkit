from typing import List
import numpy as np
import scipy


class Operation:
    def __init__(self, operator):
        self._operator = operator

    def __repr__(self):
        return self._operator

    def __call__(self, *args):
        ...


class SimpleFunction(Operation):
    def __init__(self, operator, func):
        super().__init__(operator)
        self._func = func

    def __call__(self, a):
        return exec(f"{self._operator}{a}", {self._operator: self._func})


class Arithmetic(Operation):
    def __init__(self, operator):
        super().__init__(operator)

    def __call__(self, a, b):
        return eval(f"{a} {self._operator} {b}")


default_primitives: List = [
    Arithmetic(operator) for operator in ["+", "-", "*", "/", "**"]
]
default_primitives += [
    SimpleFunction(operator[0], operator[1])
    for operator in {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "heaviside": np.heaviside,
        "expit": scipy.expit,
        }
]
