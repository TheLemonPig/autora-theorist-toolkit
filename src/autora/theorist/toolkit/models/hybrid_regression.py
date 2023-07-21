import random

import numpy as np
import tqdm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from autora.theorist.toolkit.methods.metrics import minimum_description_length
from autora.theorist.toolkit.methods.regression import regression_handler
from autora.theorist.toolkit.methods.rules import replace_node
from autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from autora.theorist.toolkit.models.tree import Tree

prior_dict_ = {
    "+": 3.0,
    "-": 3.0,
    "*": 3.0,
    "/": 3.0,
    "**": 3.0,
    "sin": 10.0,
    "exp": 10.0,
    "log": 10.0,
}
temperatures_ = [1.04**n for n in range(20)]


class HybridModel(BaseEstimator):
    def __init__(self, trees=3, sklearn_regressor=LinearRegression):
        self.reg = sklearn_regressor
        self.tree_sets = [
            [
                SymbolicRegressor(temperature, prior_dict_)
                for temperature in temperatures_
            ]
            for _ in range(trees)
        ]
        self.constants = [1.0 for _ in range(trees + 1)]
        self.mdl = np.inf
        self.prior_dict = prior_dict_
        self.model_ = Tree()

    def fit(self, X, y, epochs=1500):
        for n in tqdm(range(epochs)):
            tree_set = random.choice(self.trees)
            self.fit_step(tree_set, X, y)
            self.tree_swap(tree_set, X, y)

    @regression_handler
    def fit_step(self, tree_set, X, y):
        for tree, i in enumerate(tree_set):
            tree.step()
            if not tree.visited():
                tree._record_visit()
                self.swap(tree_set, 0, i)
                W = self.intermediate(X)
                self.reg.fit(W, y)
                mdl = minimum_description_length(
                    y_true=y,
                    y_pred=self.reg.predict(W),
                    n=X.shape[0],
                    k=len(tree.get_parameters()),
                    prior_dict=self.prior_dict,
                    expr_str=self.make_full_tree_str(),
                    temperature=tree.temperature,
                )
                self.swap(tree_set, 0, i)
                if mdl < self.mdl:
                    self.mdl = mdl
                else:
                    tree.backstep()
            else:
                tree.backstep()

    def predict(self, X):
        W = self.intermediate(X)
        return self.predict(W)

    def intermediate(self, X):
        W = np.ones((X.shape[0], 1))
        for tree_set in self.trees:
            tree = tree_set[0]
            w = tree.predict(X)
            W = np.hstack((W, w))
        return W

    def make_full_tree_str(self):
        full_str = "__a__"
        for tree in self.trees:
            full_str = str(tree.model_) + "*__a__+" + full_str
        return full_str

    def tree_swap(self, tree_set, X, y):
        j = random.choice(range(len(self.temperatures) - 1))
        self.swap(tree_set, 0, j)
        W = self.intermediate(X)
        self.reg.fit(W, y)

        mdl1 = minimum_description_length(
            y_true=y,
            y_pred=self.reg.predict(W),
            n=X.shape[0],
            k=len(tree_set[j].get_parameters()),
            prior_dict=self.prior_dict,
            expr_str=self.make_full_tree_str(),
            temperature=tree_set[j].temperature,
        )
        self.swap(tree_set, 0, j)
        self.swap(tree_set, 0, j + 1)
        W = self.intermediate(X)
        self.reg.fit(W, y)
        mdl2 = minimum_description_length(
            y_true=y,
            y_pred=self.reg.predict(W),
            n=X.shape[0],
            k=len(tree_set[j + 1].get_parameters()),
            prior_dict=self.prior_dict,
            expr_str=self.make_full_tree_str(),
            temperature=tree_set[j + 1].temperature,
        )
        self.swap(tree_set, 0, j + 1)
        mdl_change = mdl1 * (
            1 / tree_set[j + 1].temperature - 1 / tree_set[j].temperature
        ) + mdl2 * (1 / tree_set[j].temperature - 1 / tree_set[j + 1].temperature)
        if replace_node(-mdl_change):
            self.swap(tree_set, j, j + 1)
            return True
        else:
            return False

    def swap(self, tree_set, i, j):
        tree1, tree2 = tree_set[i], tree_set[j]
        tree_set[j], tree_set[i] = tree1, tree2
