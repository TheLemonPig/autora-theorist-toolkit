import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm
from inspect import signature
from typing import List
from copy import deepcopy
import pandas as pd
import random

from src.autora.theorist.toolkit.components.nodes import Variable, Parameter, Operator
from src.autora.theorist.toolkit.components.primitives import default_primitives
from src.autora.theorist.toolkit.models.tree import Tree
from src.autora.theorist.toolkit.models.memory import Stack
from src.autora.theorist.toolkit.methods.fitting import scipy_curve_fit
from src.autora.theorist.toolkit.methods.regression import regression_handler, canonical
from src.autora.theorist.toolkit.methods.rules import less_than


class SymbolicRegressor(BaseEstimator):

    # TODO: replace tree=None with expression=None once build_tree() is made
    def __init__(self, tree=None, moves=None, primitives=None,
                 metric=None):
        self.DVs = dict()  # currently unused
        self.IVs = dict()  # currently unused
        self.model_ = Tree() if tree is None else tree
        self.metric = mean_squared_error if metric is None else metric
        self._primitives = default_primitives if primitives is None else primitives
        self._expression = None
        self._value = None
        self._cache: Stack = Stack()
        self._moves: List[str] = ['Root/None', 'None/Root', 'Node/Node', 'Node/Leaf', 'Leaf/Node', 'Leaf/Leaf']\
            if moves is None else moves
        self._variables: List[str] = list()
        self._trace: List[float] = []
        self._complete(depth=0)
        self._history = []
        self._visit_list = set()
        self._error = np.inf

    def __repr__(self, N_CHAR_MAX=None):
        if N_CHAR_MAX is None:
            # Symbolic Regressor __repr__ method
            return self.model_.__repr__()
        else:
            # Sci-kit Learn __repr__ method
            super().__repr__(N_CHAR_MAX)

    def load_data(self, x, y):
        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            dv_variables, iv_variables = x.columns, y.columns
        elif isinstance(x, np.ndarray):
            dv_variables = ['_x' + str(i) + '_' for i in range(x.shape[1])]
            iv_variables = ['_y' + str(i) + '_' for i in range(y.shape[1])]
        else:
            raise TypeError('Only valid types for x and y are pandas DataFrames and Numpy nd-arrays')
        self.DVs = {dv_variables[i]: np.array(x[:, i]) for i in range(len(dv_variables))}
        self.IVs = {iv_variables[i]: np.array(y[:, i]) for i in range(len(iv_variables))}
        self._variables = dv_variables

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000,
            fitter=scipy_curve_fit, metric=mean_squared_error, accept=less_than,
            verbose=False):
        self.load_data(X, y)
        y_pred = self.predict(X)
        self._error = metric(y, y_pred)
        for _ in tqdm(range(epochs)):
            self.fit_step(X=X, y=y, fitter=fitter, metric=metric, accept=accept)
            self._history.append(str(self.model_))
            self._trace.append(self._error)
        if verbose:
            print(f'best model: {self.model_}\nError: {self._error}')
        return self

    @regression_handler
    def fit_step(self, X: np.ndarray, y: np.ndarray, move=None,
                 fitter=scipy_curve_fit, metric=mean_squared_error, accept=less_than,
                 *args, **kwargs):
        self.load_data(X, y)
        self.step(move=move)
        if not self.visited():
            self.optimize_parameters(X, y, fitter)
            y_pred = self.predict(X)
            error = metric(y, y_pred, *args, **kwargs)
            self._record_visit()
            if accept(error, self._error):
                self._error = error
            else:
                self.back_step()
        else:
            self.back_step()

    def predict(self, X, verbose=False):
        Xs = tuple(X[:, i] for i in range(X.shape[1]))
        # Xs = X
        # self._compile(x_type=type(X))
        if verbose:
            plt.plot(self._trace)
            plt.show()
        try:
            y_pred = self._expression(Xs)
            if isinstance(y_pred, float):
                y_pred = np.ones(X.shape[0]) * y_pred
            return y_pred
        except NameError:
            raise NameError(f'Why is this caused again?\n{self.model_}')

    def get_expression(self):
        return self._expression

    def step(self, move='', path=1):
        self.cache()
        moves = []
        for steps in range(path):
            if move not in self._moves:
                move = random.choice(self._moves)
            if move == 'Root/None':
                self.remove_root()
            elif move == 'None/Root':
                self.add_root()
            elif move == 'Node/Node':
                self.replace_node()
            elif move == 'Node/Leaf':
                self.replace_node_with_leaf()
            elif move == 'Leaf/Node':
                self.replace_leaf_with_node()
            elif move == 'Leaf/Leaf':
                self.replace_leaf()
            else:
                raise KeyError('Unknown move type')
            moves.append(move)
        self._compile()
        return moves

    def optimize_parameters(self, X, y, fitter):
        if len(self.model_.get_parameters()) > 0:
            fitted_parameters = fitter(X, y, self.get_expression())
            self.model_.set_parameters(fitted_parameters)
            self._compile()

    def cache(self):
        self._cache.push(deepcopy(self.model_.get_root()))

    def peek(self):
        return self._cache.top()

    def back_step(self):
        self.model_._root = self._cache.pop()
        self._compile()

    def remove_root(self):
        if self.model_.get_root().has_children():
            new_root = random.choice(self.model_.get_root().get_children())
            new_root._parent = None
            self.model_._root = new_root

    def add_root(self):
        new_root = self.make_node(random.choice(list(set(self.order(node) for node in self._primitives))))
        old_root = self.model_.get_root()
        self.model_._root = new_root
        if self.model_.get_root().has_children():
            self.replace(random.choice(self.model_.get_root().get_children()), old_root)

    def replace_node(self):
        nodes = [node for node in self.model_.get_all_nodes() if node.has_children()]
        if not self.model_.get_root().has_children():
            nodes.append(self.model_.get_root())
        node = random.choice(list(node for node in nodes))
        self.replace(node)

    def replace_node_with_leaf(self):
        nodes = [node for node in self.model_.get_all_nodes() if not node.has_children()]
        node = random.choice(list(node for node in nodes))
        self.replace(node)

    def replace_leaf(self):
        leaves = [node for node in self.model_.get_all_nodes() if not node.has_children()]
        node = random.choice(list(node for node in leaves))
        self.replace(node)

    def replace_leaf_with_node(self):
        leaves = [node for node in self.model_.get_all_nodes() if node.has_children()]
        if not self.model_.get_root().has_children():
            leaves.append(self.model_.get_root())
        node = random.choice(list(node for node in leaves))
        self.replace(node)

    def replace(self, old_node, new_node=None):
        if new_node is None:
            new_node = self.make_node(self.order(old_node))
        self.model_.replace(old_node, new_node)
        self._complete()

    def make_node(self, order=0):
        filtered_primitives = self._filter_primitives(order)
        if order == 0:
            node_types = [Parameter()]
            # If there are variables (i.e. always except during __init__) add variables
            if len(self._variables) > 0:
                node_types += [Variable(value=random.choice(list(self.DVs.keys())))]
        else:
            node_types = [Operator(value=random.choice(filtered_primitives))]
        # assert len(weights) == len(node_types), 'number of weights must correspond to number of node types'
        return random.choices(node_types, k=1)[0]

    def _complete(self, depth=0):
        while not self.model_.is_filled():
            for node in self.model_.get_uninitialized():
                new_node = self.make_node(order=depth)
                self.model_.replace(node, new_node)
            depth -= 1
        self._compile()

    def _compile(self, x_type=tuple):
        self.model_.catalog()
        str_repr = self._sub_in_x(self.model_.__repr__(), x_type=x_type)
        func_str = 'def func(X,' + ','.join(
            [item[0] + '=' + str(item[1]) for item in self.model_.get_parameter_dict().items()]) + '): '
        func_str += 'return ' + str_repr
        namespace = {str(parameter): parameter.get_value() for parameter in self.model_.get_parameters()}
        exec(func_str, {'np': np}, namespace)
        self._expression = namespace['func']
        return 'func', namespace

    def visited(self):
        return canonical(str(self.model_)) in self._visit_list

    def get_parameters(self):
        return self.model_.get_parameters()

    def _record_visit(self):
        self._visit_list.add(canonical(str(self.model_)))

    def _filter_primitives(self, order):
        return [primitive for primitive in self._primitives
                if len(signature(primitive).parameters)
                == order]

    @staticmethod
    def order(node):
        return len(signature(node).parameters)

    @staticmethod
    def _sub_in_x(str_repr, x_type=tuple):
        while '_x' in str_repr:
            start = str_repr.index('_x')
            stop = str_repr.index('_', start + 1)
            num = str_repr[start + 2:stop]
            if x_type is tuple:
                str_repr = str_repr[:start] + 'X[' + num + ']' + str_repr[stop + 1:]
            elif x_type is np.ndarray:
                str_repr = str_repr[:start] + 'X[,:' + num + ']' + str_repr[stop + 1:]
            else:
                raise ValueError('Invalid data-type for X-variable')
        return str_repr


if __name__ == '__main__':
    model = SymbolicRegressor()
    for _ in range(30):
        model.step()
    print(model.model_)
