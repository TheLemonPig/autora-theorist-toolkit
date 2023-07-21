class HierarchicalSymbolicRegressor(SymbolicRegressor):

    def __init__(self, primitives=None):
        super().__init__(primitives=primitives)
        self.ids = list()
        self.id_parameters = dict()
        self.parameter_cache = Stack()

    def load_data(self, x, y, g):
        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            dv_variables, iv_variables = x.columns, y.columns
        elif isinstance(x, np.ndarray):
            dv_variables = ['_x' + str(i) + '_' for i in range(x.shape[1])]
            iv_variables = ['_y' + str(i) + '_' for i in range(y.shape[1])]
        else:
            raise TypeError('Only valid types for x and y are pandas DataFrames and Numpy nd-arrays')
        self.DVs = {dv_variables[i]: np.array(x[:, i]) for i in range(len(dv_variables))}
        self.IVs = {iv_variables[i]: np.array(y[:, i]) for i in range(len(iv_variables))}
        self.ids = np.unique(g)
        self.id_parameters = dict.fromkeys(self.ids, [0.])
        self._variables = dv_variables

    def cache(self):
        self.parameter_cache.push(deepcopy(self.id_parameters))
        super().cache()

    def back_step(self):
        self.id_parameters = self.parameter_cache.pop()
        super().back_step()

    def get_id_data(self, g, id, x, y=None):
        id_x = x[g == id]
        if len(id_x.shape) == 1:
            id_x = id_x.reshape((-1, 1))
        if y is None:
            id_y = None
        else:
            id_y = y[g == id]
            if len(id_y.shape) == 1:
                id_y = id_y.reshape((-1, 1))
        return id_x, id_y

    def update_dict(self, id, parameters):
        self.id_parameters.update({id: parameters})

    @regression_handler
    def hierarchical_fit_step(self, X, y, g,
                              fitter=scipy_curve_fit, metric=mean_squared_error, accept=less_than, *args, **kwargs):
        self.step()
        if not self.visited():
            self.optimize_parameters(X, y, g, fitter)
            self._record_visit()
            y_pred = self.predict(X, g)
            error = metric(y, y_pred, *args, **kwargs)
            if accept(error, self._error):
                self._error = error
            else:
                self.back_step()
        else:
            self.back_step()
        return True

    def predict(self, X, g):
        y_predict = []
        for id in np.unique(g):
            id_x, _ = self.get_id_data(g, id, X, None)
            parameters = self.id_parameters[id]
            self.model_.set_parameters(parameters)
            self._compile()
            y_predict += super().predict(id_x).tolist()
        return np.array(y_predict).reshape((-1, 1))

    def optimize_parameters(self, x, y, g, fitter):
        for id in self.ids:
            if len(self.model_.get_parameters()) > 0:
                id_x, id_y = self.get_id_data(g, id, x, y)
                fitted_parameters = fitter(id_x, id_y, self.get_expression())
                self.update_dict(id, fitted_parameters)
            else:
                self.update_dict(id, [])
        self._compile()
