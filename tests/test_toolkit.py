from src.autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from src.autora.theorist.toolkit.models.parallel_symbolic_regression import ParallelSymbolicRegressor
from src.autora.theorist.toolkit.models.bayesian_symbolic_regression import BayesianSymbolicRegressor
from src.autora.theorist.toolkit.models.bayesian_machine_scientist import BayesianMachineScientist
from src.autora.theorist.toolkit.models.tree import Tree
from src.autora.theorist.toolkit.models.memory import Stack
import numpy as np


def test_symbolic_regression_initialization():
    theorist = SymbolicRegressor()
    assert theorist is not None
    assert isinstance(theorist.model_, Tree)
    assert isinstance(theorist._cache, Stack)
    assert str(theorist) == '__a0__'
    assert isinstance(theorist.predict(np.ones((100, 1))), np.ndarray)


def test_bayesian_symbolic_regression_initialization():
    prior_dict_ = {'+': 1.0}
    theorist = BayesianSymbolicRegressor(prior_dict=prior_dict_)
    assert theorist is not None
    assert isinstance(theorist.model_, Tree)
    assert isinstance(theorist._cache, Stack)
    assert str(theorist) == '__a0__'
    assert isinstance(theorist.predict(np.ones((100, 1))), np.ndarray)


def test_parallel_symbolic_regression_initialization():
    prior_dict_ = {'+': 1.0}
    temperatures_ = [1.04 ** t for t in range(20)]
    theorist = ParallelSymbolicRegressor(temperatures=temperatures_, prior_dict=prior_dict_)
    assert theorist is not None


def test_bayesian_machine_scientist_initialization():
    prior_dict_ = {'+': 1.0}
    temperatures_ = [1.04 ** t for t in range(20)]
    theorist = BayesianMachineScientist(temperatures=temperatures_, prior_dict=prior_dict_)
    assert theorist is not None


if __name__ == "__main__":
    test_symbolic_regression_initialization()
    test_bayesian_symbolic_regression_initialization()
    test_parallel_symbolic_regression_initialization()
    test_bayesian_machine_scientist_initialization()
