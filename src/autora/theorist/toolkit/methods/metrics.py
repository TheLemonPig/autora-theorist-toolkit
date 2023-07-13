import numpy as np
from sklearn.metrics import mean_squared_error
from src.autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor


def continuous_log_loss(mse):
    return np.log(mse)


def continuous_bayesian_information_criterion(y_true, y_pred, n, k):
    # BIC = k * log(n) + n * (log(2pi) + LogLoss + 1) - Guimera 2020
    mse = mean_squared_error(y_true, y_pred)
    return k * np.log(n) + n * (np.log(2 * np.pi) + np.log(mse) + 1)


def priors(prior_dict, expr_str):
    prior = 0
    for k in prior_dict.keys():
        prior += prior_dict[k] * expr_str.count(k)
    return prior


def minimum_description_length(y_true, y_pred, n, k, prior_dict, expr_str, temperature=1.0):
    bic = continuous_bayesian_information_criterion(y_true, y_pred, n, k)
    prior = priors(prior_dict, expr_str)
    return bic/2/temperature + prior


class Loss:

    def __init__(self):
        ...

    def __call__(self, y_true, y_pred) -> float:
        ...


class MinimumDescriptionLength:

    def __init__(self, n, k, prior_dict, expr_str, bic_temp=1, prior_temp=1):
        super().__init__()
        self.n = n
        self.k = k
        self.prior_dict = prior_dict
        self.expr_str = expr_str
        self.bic_temp = bic_temp
        self.prior_temp = prior_temp

    def __call__(self, y_true, y_pred) -> float:
        return minimum_description_length(y_true, y_pred, self.n, self.k, self.prior_dict, self.expr_str)


class MDLChange:

    def __init__(self, regressor: SymbolicRegressor, prior_dict, initial_mdl):
        self.regressor = regressor
        self.current_mdl = initial_mdl
        self.prior_dict = prior_dict

    def __call__(self, y_true, y_pred):
        n = len(y_pred)
        k = len(self.regressor.model_.get_parameters())
        expr_str = str(self.regressor.model_)
        new_mdl = minimum_description_length(y_true, y_pred, n, k, self.prior_dict, expr_str)
        mdl_change = new_mdl - self.current_mdl
        self.current_mdl = new_mdl
        return mdl_change

# class MDLChange(MDLLoss):
#
#     def __init__(self, n_old, n_new, k_old, k_new, prior_dict, expr_str_old, expr_str_new, bic_temp=1, prior_temp=1):
#         self.old = super().__init__(n_old, k_old, prior_dict, expr_str_old, bic_temp, prior_temp)
#         self.new = super().__init__(n_new, k_new, prior_dict, expr_str_new, bic_temp, prior_temp)
#
#     def __call__(self, y_true, y_pred) -> float:
#         return self.new(y_true, y_pred) - self.old(y_true, y_pred)
