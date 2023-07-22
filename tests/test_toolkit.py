import numpy as np
import scipy

from autora.theorist.toolkit.models.bayesian_machine_scientist import (
    BayesianMachineScientist,
)
from autora.theorist.toolkit.models.bayesian_symbolic_regression import (
    BayesianSymbolicRegressor,
)
from autora.theorist.toolkit.models.hierarchical_bayesian_symbolic_regression import (
    HierarchicalBayesianSymbolicRegression,
)
from autora.theorist.toolkit.models.memory import Stack
from autora.theorist.toolkit.models.parallel_symbolic_regression import (
    ParallelSymbolicRegressor,
)
from autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from autora.theorist.toolkit.models.tree import Tree


def test_symbolic_regression_initialization():
    theorist = SymbolicRegressor()
    assert theorist is not None
    assert isinstance(theorist.model_, Tree)
    assert isinstance(theorist._cache, Stack)
    assert str(theorist) == "__a0__"
    assert isinstance(theorist.predict(np.ones((100, 1))), np.ndarray)


def test_bayesian_symbolic_regression_initialization():
    prior_dict_ = {"+": 1.0}
    theorist = BayesianSymbolicRegressor(prior_dict=prior_dict_)
    assert theorist is not None
    assert isinstance(theorist.model_, Tree)
    assert isinstance(theorist._cache, Stack)
    assert str(theorist) == "__a0__"
    assert isinstance(theorist.predict(np.ones((100, 1))), np.ndarray)


def test_parallel_symbolic_regression_initialization():
    prior_dict_ = {"+": 1.0}
    temperatures_ = [1.04**t for t in range(20)]
    theorist = ParallelSymbolicRegressor(
        temperatures=temperatures_, prior_dict=prior_dict_
    )
    assert theorist is not None


def test_bayesian_machine_scientist_initialization():
    theorist = BayesianMachineScientist()
    assert theorist is not None


def test_hsbr_prior_restriction_and_fitting():
    prior_dict_ = {"+": 1.0, "expit": 1.0}
    hsbr = HierarchicalBayesianSymbolicRegression(prior_dict=prior_dict_)
    assert len(hsbr.theorists[-1]._primitives) > 0
    x = scipy.special.expit(np.linspace(0, 1, 100)).reshape((-1, 1))
    y = 1 + x
    g = np.ones_like(x)
    hsbr.fit(x, y, g, epochs=30)
    assert "-" not in hsbr.theorists[-1].model_
    assert "*" not in hsbr.theorists[-1].model_
    assert "**" not in hsbr.theorists[-1].model_
    assert "/" not in hsbr.theorists[-1].model_
    assert "cos" not in hsbr.theorists[-1].model_
    assert "exp" not in hsbr.theorists[-1].model_
    assert "log" not in hsbr.theorists[-1].model_
    assert "sin" not in hsbr.theorists[-1].model_
    assert (
        np.sum(hsbr.theorists[-1].predict(x, g) - y) == 0
    ), f"{hsbr.theorists[-1].model_} but should be 1+expit(x)"


if __name__ == "__main__":
    test_symbolic_regression_initialization()
    test_bayesian_symbolic_regression_initialization()
    test_parallel_symbolic_regression_initialization()
    test_bayesian_machine_scientist_initialization()
