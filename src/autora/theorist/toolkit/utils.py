import random
from src.autora.theorist.toolkit.methods.fitting import FittingError
import warnings
import functools
from sympy import sympify

##################
#       Functions
##################


def canonical(expr_str: str):
    expr_str = expr_str.replace('np.', '')
    expr_sym = sympify(expr_str)
    expr_str = str(expr_sym)
    while '__a' in expr_str:
        start = expr_str.index('__a')
        stop = expr_str.index('__', start+1) + 2
        expr_str = expr_str[:start]+'__c__'+expr_str[stop:]
    return expr_str


# function to build Tree from Expression
# def build_tree():
#     ...

# def parse_expr_str(expr_str):
#     parsed_expr = []


###
# decision_sample
###
# takes: takes a list of lists of floats corresponding to weights for choices
# gives: list of integers, with each integer corresponding to the index chosen for each list of floats given
def decision_sample(search_space):
    sample = []
    for options in search_space:
        sample.append(random.choices(range(len(options)), weights=options, k=1))
    return sample


###
# (hamiltonian) mcmc sample
###
# takes: model, search space with markov chain weights, number of hamiltonian steps (probability for bernoulli dist)
# gives: new model
# def mcmc_sample():
#    ...


def regression_handler(func):
    @functools.wraps(func)
    def handler(obj, *args, **kwargs):
        assert hasattr(obj, 'back_step'), 'Regression Handler assumes back_step method'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                func(obj, *args, **kwargs)
            except (FittingError, TypeError, ValueError, ZeroDivisionError):
                obj.back_step()
    return handler
