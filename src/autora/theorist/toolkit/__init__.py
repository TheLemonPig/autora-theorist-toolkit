"""
Example Theorist
"""


from sklearn.base import BaseEstimator


class ExampleRegressor(BaseEstimator):
    """
    Include inline mathematics in docstring \\(x < 1\\) or $c = 3$
    or block mathematics:

    \\[
        x + 1 = 3
    \\]


    $$
    y + 1 = 4
    $$

    """

    def __init__(self):
        pass

    def fit(self, conditions, observations):
        pass

    def predict(self, conditions):
        pass
