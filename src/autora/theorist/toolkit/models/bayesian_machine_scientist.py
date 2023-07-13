import numpy as np
import logging
from tqdm import tqdm
from src.autora.theorist.toolkit.components.primitives import Arithmetic, SimpleFunction
from src.autora.theorist.toolkit.methods.rules import replace_node
from src.autora.theorist.toolkit.models.bayesian_symbolic_regression import BayesianSymbolicRegressor
from src.autora.theorist.toolkit.methods.metrics import minimum_description_length
from src.autora.theorist.toolkit.methods.regression import clean_equation
from sklearn.metrics import mean_squared_error
import random

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class BayesianMachineScientist:

    def __init__(self, temperatures, prior_dict):
        self.temperatures = temperatures
        self.prior_dict = prior_dict
        self.theorists = [BayesianSymbolicRegressor(prior_dict=prior_dict) for _ in self.temperatures]

    def fit(self, x, y, epochs=100):
        n_swaps = 0
        # running fit for each theorist one at a time before trying a swap and then continuing
        for n in tqdm(range(epochs)):
            for i, theorist in enumerate(self.theorists):
                theorist.load_data(x, y)
                theorist.mcmc_step(x, y)
                _logger.debug("Finish iteration {}".format(n))
            n_swaps += int(self.tree_swap(x, y))
        print(f'Number of Tree Swaps: {n_swaps} out of {epochs} epochs')

        for theorist in self.theorists:
            metric = mean_squared_error(y, theorist.predict(X=x))
            print(f'best model: {clean_equation(str(theorist.model_))}\nError: {metric}')
            param_list = [param[0] + ':' + str(param[1]) for param in theorist.model_.get_parameter_dict().items()]
            print(', '.join(param_list))

    # function for tree swapping parallel temperatures
    def tree_swap(self, x, y):
        idx = random.choice(range(len(self.temperatures)-1))
        temp1, temp2 = self.temperatures[idx:idx+2]
        theorist1, theorist2 = self.theorists[idx:idx+2]
        y_pred1 = theorist1.predict(x)
        if isinstance(y_pred1, float):
            y_pred1 = np.ones(y.shape) * y_pred1
        y_pred2 = theorist2.predict(x)
        if isinstance(y_pred2, float):
            y_pred2 = np.ones(y.shape) * y_pred2
        n = x.shape[0]
        k1, k2 = len(theorist1.get_parameters()), len(theorist2.get_parameters())
        expr_str1, expr_str2 = str(theorist1.model_), str(theorist1.model_)
        mdl1 = minimum_description_length(y, y_pred1, n, k1, self.prior_dict, expr_str1)
        mdl2 = minimum_description_length(y, y_pred2, n, k2, self.prior_dict, expr_str2)
        mdl_change = mdl1*(1/temp2-1/temp1) + mdl2*(1/temp1-1/temp2)
        if replace_node(-mdl_change):
            self.theorists[idx:idx+2] = theorist2, theorist1
            return True
        else:
            return False


if __name__ == '__main__':
    # create synthetic data
    x1 = np.linspace(1, 5, 100).reshape(-1, 1)
    x2 = np.linspace(7, 2, 100).reshape(-1, 1)
    y_ = x1 ** 2 + x1 + 1
    x_ = np.hstack((x1, x2))

    # initialize regressor
    primitives_ = [
        Arithmetic(operator) for operator in ['+', '-', '*', '/', '**']
    ]
    primitives_ += [SimpleFunction(operator) for operator in ['np.sin', 'np.cos', 'np.exp', 'np.log']]
    epochs_ = 300
    temperatures_ = [1.04 ** n for n in range(20)]
    prior_dict_ = {'+': 3.0,
                   '-': 3.0,
                   '*': 3.0,
                   '/': 3.0,
                   '**': 3.0,
                   'sin': 10.0,
                   'exp': 10.0,
                   'log': 10.0}

    # initialize theorist and fit
    bms = BayesianMachineScientist(temperatures=temperatures_, prior_dict=prior_dict_)
    bms.fit(x_, y_, epochs=epochs_)
