import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error
from autora.theorist.toolkit.components.primitives import Arithmetic, SimpleFunction
from autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from autora.theorist.toolkit.methods.metrics import MinimumDescriptionLength
from autora.theorist.toolkit.methods.regression import regression_handler
from autora.theorist.toolkit.methods.fitting import scipy_curve_fit
from autora.theorist.toolkit.methods.rules import replace_node, less_than


class HierarchicalBayesianSymbolicRegression:

  def __init__(self, temperatures, prior_dict):

    self.temperatures = temperatures
    self.prior_dict = prior_dict
    self.theorists = [HierarchicalSymbolicRegressor() for _ in self.temperatures]

  def fit(self, x, y, g, epochs=100):
    n_swaps = 0
    for n in tqdm(range(epochs)):
        for i, theorist in enumerate(self.theorists):
            metric = MinimumDescriptionLength(n=x.shape[0], k=len(theorist.model_.get_parameters()),
                                              prior_dict=self.prior_dict, expr_str=str(theorist.model_),
                                              bic_temp=self.temperatures[i])
            theorist.hierarchical_fit_step(X=x, y=y,g=g, metric=metric)
            _logger.debug("Finish iteration {}".format(n))
        self.tree_swap(x, y)

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