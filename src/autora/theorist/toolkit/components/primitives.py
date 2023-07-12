
class Operator:

    def __init__(self, operator):
        self._operator = operator

    def __repr__(self):
        return self._operator

    def __call__(self, *args):
        ...


class SimpleFunction(Operator):

    def __init__(self, operator):
        super().__init__(operator)

    def __call__(self, a):
        return exec(f'{self._operator}{a}')


class Arithmetic(Operator):

    def __init__(self, operator):
        super().__init__(operator)

    def __call__(self, a, b):
        return eval(f'{a} {self._operator} {b}')


default_primitives = [
    Arithmetic(operator) for operator in ['+', '-', '*', '/', '**']
]
default_primitives += [SimpleFunction(operator) for operator in ['np.sin', 'np.cos', 'np.exp', 'np.log']]
