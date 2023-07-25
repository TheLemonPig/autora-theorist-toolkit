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


def test_hbsr_prior_restriction_and_fitting():
    prior_dict_ = {"+": 1.0, "expit": 1.0}
    hbsr = HierarchicalBayesianSymbolicRegression(prior_dict=prior_dict_)
    assert len(hbsr.theorists[-1]._primitives) > 0
    x = scipy.special.expit(np.linspace(0, 1, 100)).reshape((-1, 1))
    y = 1 + x
    g = np.ones_like(x)
    hbsr.fit(x, y, g, epochs=30)
    assert "-" not in hbsr.theorists[-1].model_
    assert "*" not in hbsr.theorists[-1].model_
    assert "**" not in hbsr.theorists[-1].model_
    assert "/" not in hbsr.theorists[-1].model_
    assert "cos" not in hbsr.theorists[-1].model_
    assert "exp" not in hbsr.theorists[-1].model_
    assert "log" not in hbsr.theorists[-1].model_
    assert "sin" not in hbsr.theorists[-1].model_
    assert (
        np.sum(hbsr.theorists[-1].predict(x, g) - y) == 0
    ), f"{hbsr.theorists[-1].model_} but should be 1+expit(x)"


def test_hbsr_multi_x():
    x = np.linspace(0, 1, 200).reshape((-1, 2))
    y = 1 + x[:, 0] + x[:, 1]
    y = y.reshape((-1, 1))
    g = np.ones_like(y)
    hbsr = HierarchicalBayesianSymbolicRegression()
    hbsr.fit(x=x, y=y, g=g, epochs=30)


def test_hbsr_seed():
    hbsr1 = HierarchicalBayesianSymbolicRegression(seed=1)
    hbsr2 = HierarchicalBayesianSymbolicRegression(seed=2)
    for _ in range(25):
        hbsr1.theorists[-1].step()
        hbsr2.theorists[-1].step()
    assert str(hbsr1.theorists[-1].model_) != str(
        hbsr2.theorists[-1].model_
    ), f"{str(hbsr1.theorists[-1].model_)} {str(hbsr2.theorists[-1].model_)}"


def test_hbsr_model_():
    hbsr = HierarchicalBayesianSymbolicRegression()
    assert hasattr(hbsr, "model_") and hbsr.model_ is not None


if __name__ == "__main__":
    test_symbolic_regression_initialization()
    test_bayesian_symbolic_regression_initialization()
    test_parallel_symbolic_regression_initialization()
    test_bayesian_machine_scientist_initialization()
    test_hbsr_multi_x()
