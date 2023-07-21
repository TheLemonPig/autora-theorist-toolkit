import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error

from autora.theorist.toolkit.methods.fitting import scipy_curve_fit
from autora.theorist.toolkit.methods.rules import less_than
from autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor


class HierarchicalSymbolicRegressor(SymbolicRegressor):
    def __init__(self):
        super().__init__()
        self.ids = list()
        self.id_parameters = dict()

    def load_data(self, x, y, g):
        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            dv_variables, iv_variables = x.columns, y.columns
        elif isinstance(x, np.ndarray):
            dv_variables = ["_x" + str(i) + "_" for i in range(x.shape[1])]
            iv_variables = ["_y" + str(i) + "_" for i in range(y.shape[1])]
        else:
            raise TypeError(
                "Only valid types for x and y are pandas DataFrames and Numpy nd-arrays"
            )
        self.DVs = {
            dv_variables[i]: np.array(x[:, i]) for i in range(len(dv_variables))
        }
        self.IVs = {
            iv_variables[i]: np.array(y[:, i]) for i in range(len(iv_variables))
        }
        self.ids = np.unique(g)
        self.id_parameters = dict.fromkeys(self.ids, None)
        self._variables = dv_variables

    def get_id_data(self, x, y, g, id):
        x, y = x[g == id], y[g == id]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            x = y.reshape(-1, 1)
        return x, y

    def update_dict(self, id, parameters):
        self.id_parameters.update({id: parameters})

    def hierarchical_fit_step(
        self,
        X,
        y,
        g,
        fitter=scipy_curve_fit,
        metric=mean_squared_error,
        accept=less_than,
        *args,
        **kwargs,
    ):
        self.load_data(X, y, g)
        self.step()
        if not self.visited():
            self.optimize_parameters(X, y, g, fitter)
            self._record_visit()
            y_pred = self.predict(X)
            error = metric(y, y_pred, *args, **kwargs)
            if accept(error, self._error):
                self._error = error
            else:
                self.back_step()
        else:
            self.back_step()

    def optimize_parameters(self, x, y, g, fitter):
        if len(self.model_.get_parameters()) > 0:
            for id in self.ids:
                id_x, id_y = self.get_id_data(x, y, g, id)
                fitted_parameters = fitter(id_x, id_y, self.get_expression())
                self.update_dict(id, fitted_parameters)
                print(f"{id}")
                self._hier_compile()

    def _hier_compile(self, x_type=tuple):
        self.model_.catalog()
        str_repr = self._sub_in_x(self.model_.__repr__(), x_type=x_type)
        func_str = "def func(X,id):" + "\n"
        func_str += "\t" + ",".join(
            str(parameter) for parameter in self.model_.get_parameters()
        )
        func_str += "\t" + "= dic[id]" + "\n"
        func_str += "\t" + "return " + str_repr
        namespace = {}
        exec(func_str, {"np": np, "scipy": scipy, "dic": self.id_parameters}, namespace)
        self._expression = namespace["func"]
        return "func", namespace
