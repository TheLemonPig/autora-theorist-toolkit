from src.autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
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


# Note: We encourage you to adjust this test and write more tests.

if __name__ == "__main__":
    test_symbolic_regression_initialization()
