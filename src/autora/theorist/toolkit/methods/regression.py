import functools
import warnings
from src.autora.theorist.toolkit.methods.fitting import FittingError
from sympy import sympify
from scipy.optimize import OptimizeWarning
import functools


def regression_handler(func):
    @functools.wraps(func)
    def handler(obj, *args, **kwargs):
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        assert hasattr(obj, 'back_step'), 'Regression Handler assumes back_step method'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                func(obj, *args, **kwargs)
            except (FittingError, TypeError, ValueError, ZeroDivisionError):
                obj.back_step()
    return handler


def canonical(expr_str: str):
    expr_str = expr_str.replace('np.', '')
    expr_sym = sympify(expr_str)
    expr_str = str(expr_sym)
    while '__a' in expr_str:
        start = expr_str.index('__a')
        stop = expr_str.index('__', start+1) + 2
        expr_str = expr_str[:start]+'__c__'+expr_str[stop:]
    return expr_str

def clean_equation(expr_str):
    expr_str = expr_str.replace('np.', '')
    expr_sym = sympify(expr_str)
    return str(expr_sym)
