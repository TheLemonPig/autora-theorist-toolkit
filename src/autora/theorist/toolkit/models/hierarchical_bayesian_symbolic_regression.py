import logging
import random

import numpy as np
from tqdm import tqdm

from autora.theorist.toolkit.components.primitives import default_primitives
from autora.theorist.toolkit.methods.metrics import MinimumDescriptionLength
from autora.theorist.toolkit.methods.rules import replace_node
from autora.theorist.toolkit.models.hierarchical_symbolic_regressor import (
    HierarchicalSymbolicRegressor,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

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


class HierarchicalBayesianSymbolicRegression:
    def __init__(self, temperatures=None, prior_dict=None, primitives=None):
        self.temperatures = temperatures_ if temperatures is None else temperatures
        self.prior_dict = prior_dict_ if prior_dict is None else prior_dict
        primitives_ = default_primitives if primitives is None else primitives
        self.primitives = [
            primitive
            for primitive in primitives_
            if str(primitive) in list(self.prior_dict.keys())
        ]
        self.theorists = [
            HierarchicalSymbolicRegressor(primitives=self.primitives)
            for _ in self.temperatures
        ]

    def fit(self, x, y, g, epochs=100, verbose=False):
        n_swaps = 0
        for theorist in self.theorists:
            theorist.load_data(x, y, g)
        for n in tqdm(range(epochs)):
            for i, theorist in enumerate(self.theorists):
                metric = MinimumDescriptionLength(
                    n=x.shape[0],
                    k=len(theorist.model_.get_parameters()),
                    prior_dict=self.prior_dict,
                    expr_str=str(theorist.model_),
                    bic_temp=self.temperatures[i],
                )
                theorist.hierarchical_fit_step(X=x, y=y, g=g, metric=metric)
                _logger.debug("Finish iteration {}".format(n))
            n_swaps += int(self.tree_swap(x, y, g))
        if verbose:
            print(f"Number of tree swaps: {n_swaps} swaps out of {epochs} epochs")

    def tree_swap(self, x, y, g):
        j = random.choice(range(len(self.temperatures) - 1))
        temp1, temp2 = self.temperatures[j : j + 2]
        theorist1, theorist2 = self.theorists[j : j + 2]
        y_pred1 = theorist1.predict(x, g)
        if isinstance(y_pred1, float):
            y_pred1 = np.ones(y.shape) * y_pred1
        y_pred2 = theorist2.predict(x, g)
        if isinstance(y_pred2, float):
            y_pred2 = np.ones(y.shape) * y_pred2
        loss1 = MinimumDescriptionLength(
            n=x.shape[0],
            k=len(theorist1.model_.get_parameters()),
            prior_dict=self.prior_dict,
            expr_str=str(theorist1.model_),
            bic_temp=temp1,
        )(y, y_pred1)
        loss2 = MinimumDescriptionLength(
            n=x.shape[0],
            k=len(theorist2.model_.get_parameters()),
            prior_dict=self.prior_dict,
            expr_str=str(theorist2.model_),
            bic_temp=temp2,
        )(y, y_pred2)
        mdl_change = loss1 * (1 / temp2 - 1 / temp1) + loss2 * (1 / temp1 - 1 / temp2)
        if replace_node(-mdl_change):
            self.theorists[j : j + 2] = theorist2, theorist1
            return True
        else:
            return False
