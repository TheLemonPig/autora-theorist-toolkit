from sklearn.utils.validation import assert_all_finite
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from typing import Callable
import warnings


###
# scipy_curve_fit
###
# takes: model [Callable], x [pandas/numpy], y [pandas/numpy]
# uses: scipy curve_fit function
# gives: optimal coefficients
def scipy_curve_fit(X: np.ndarray, y: np.ndarray, expr_func: Callable):
    Xs = tuple(X[:, i] for i in range(X.shape[1]))
    y = np.ravel(y)
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        fitted_parameters = curve_fit(expr_func, Xs, y)[0]
        assert_all_finite(fitted_parameters)
        return fitted_parameters
    except (ValueError, ZeroDivisionError, TypeError, RuntimeError) as err:
        raise FittingError(err, 'scipy\'s curve fit failed')


def fit_parameters(X: np.ndarray, y: np.ndarray, expr_func, fit_func=scipy_curve_fit):
    return fit_func(X, y, expr_func)


class FittingError(Exception):

    def __init__(self, err, message=''):
        message += '\n'+str(err)
        super().__init__(message)
