import numpy as np
import logging
from tqdm import tqdm
import random
from src.autora.theorist.toolkit.components.primitives import Arithmetic, SimpleFunction
from src.autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from src.autora.theorist.toolkit.methods.metrics import MinimumDescriptionLength
from src.autora.theorist.toolkit.methods.rules import replace_node

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class ParallelSymbolicRegressor:

    def __init__(self, temperatures, prior_dict):

        self.temperatures = temperatures
        self.prior_dict = prior_dict
        self.theorists = [SymbolicRegressor() for _ in self.temperatures]

    def fit(self, x, y):
        n_swaps = 0
        for n in tqdm(range(epochs)):
            for i, theorist in enumerate(self.theorists):
                metric = MinimumDescriptionLength(n=x.shape[0], k=len(theorist.model_.get_parameters()),
                                                  prior_dict=self.prior_dict, expr_str=str(theorist.model_),
                                                  bic_temp=self.temperatures[i])
                theorist.fit_step(X=x, y=y, metric=metric)
                _logger.debug("Finish iteration {}".format(n))
            n_swaps += int(self.tree_swap(x, y))
        print(f'Number of Tree Swaps: {n_swaps} out of {epochs} epochs')

        for theorist in self.theorists:
            loss = MinimumDescriptionLength(n=x.shape[0], k=len(theorist.model_.get_parameters()),
                                            prior_dict=self.prior_dict, expr_str=str(theorist.model_),
                                            bic_temp=self.temperatures[0])
            print(f'best model: {theorist.model_}\nError: {loss(y, theorist.predict(x))}')
            param_list = [param[0]+':'+str(param[1]) for param in theorist.model_.get_parameter_dict().items()]
            print(', '.join(param_list))

    def tree_swap(self, x, y):
        j = random.choice(range(len(self.temperatures)-1))
        temp1, temp2 = self.temperatures[j:j+2]
        theorist1, theorist2 = self.theorists[j:j+2]
        y_pred1 = theorist1.predict(x)
        if isinstance(y_pred1, float):
            y_pred1 = np.ones(y.shape) * y_pred1
        y_pred2 = theorist2.predict(x)
        if isinstance(y_pred2, float):
            y_pred2 = np.ones(y.shape) * y_pred2
        loss1 = MinimumDescriptionLength(n=x.shape[0], k=len(theorist1.model_.get_parameters()),
                                         prior_dict=self.prior_dict, expr_str=str(theorist1.model_),
                                         bic_temp=temp1)(y, y_pred1)
        loss2 = MinimumDescriptionLength(n=x.shape[0], k=len(theorist2.model_.get_parameters()),
                                         prior_dict=self.prior_dict, expr_str=str(theorist2.model_),
                                         bic_temp=temp2)(y, y_pred2)
        mdl_change = loss1*(1/temp2-1/temp1) + loss2*(1/temp1-1/temp2)
        if replace_node(-mdl_change):
            self.theorists[j:j+2] = theorist2, theorist1
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
    primitives = [
        Arithmetic(operator) for operator in ['+', '-', '*', '/', '**']
    ]
    primitives += [SimpleFunction(operator) for operator in ['np.sin', 'np.cos', 'np.exp', 'np.log']]
    epochs = 300
    temps = [1.04 ** n for n in range(20)]
    # temperatures = [1.0]
    prior_dict_ = {'+': 10.0,
                   '-': 3.0,
                   '*': 10.0,
                   '/': 3.0,
                   '**': 10.0,
                   'sin': 0.01,
                   'exp': 0.01,
                   'log': 0.01}
    bsr = ParallelSymbolicRegressor(temperatures=temps, prior_dict=prior_dict_)
    bsr.fit(x_, y_)

