from src.autora.theorist.toolkit.models.symbolic_regression import SymbolicRegressor
from src.autora.theorist.toolkit.methods.rules import add_root, remove_root, replace_node
from src.autora.theorist.toolkit.methods.metrics import minimum_description_length
from src.autora.theorist.toolkit.methods.fitting import scipy_curve_fit
from src.autora.theorist.toolkit.methods.regression import regression_handler
import random


class BayesianSymbolicRegressor(SymbolicRegressor):

    def __init__(self, prior_dict, temperature=1.0, tree=None, moves=None):
        super().__init__(tree, moves)
        self.temperature = temperature
        self.prior_dict = prior_dict

    @regression_handler
    def mcmc_step(self, X, y, fitter=scipy_curve_fit):
        move = random.choice(self._moves)
        kwargs = self.get_stats(move, X, y)

        self.step(move=move)
        if not self.visited():
            self.optimize_parameters(X, y, fitter)
            kwargs['new_mdl'] = minimum_description_length(y_true=y, y_pred=self.predict(X),
                                                           n=X.shape[0], k=len(self.get_parameters()),
                                                           prior_dict=self.prior_dict, expr_str=str(self.model_),
                                                           temperature=self.temperature)
            # mse = mean_squared_error(y, self.predict(X))
            # print(f"model: {str(self.model_)}, mse: {mse}, mdl: {kwargs['new_mdl']}")
            self._record_visit()
            if self.mcmc_ruling(move, **kwargs):
                self._error = kwargs['new_mdl']
            else:
                self.back_step()
        else:
            self.back_step()

    # return relevant stats for this mcmc move
    def get_stats(self, move, X, y):
        stats = dict()
        stats['X'], stats['y'] = X, y
        stats['expr_str'] = str(self.model_)
        stats['old_mdl'] = minimum_description_length(y_true=y, y_pred=self.predict(X),
                                                      n=X.shape[0], k=len(self.get_parameters()),
                                                      prior_dict=self.prior_dict, expr_str=str(self.model_),
                                                      temperature=self.temperature)
        if move == 'None/Root':
            stats['num_rr'] = self.get_num_root_options()
        elif move == 'Root/None':
            pass
        elif move == 'Node/Node':
            pass
        elif move == 'Node/Leaf':
            pass
        elif move == 'Leaf/Node':
            pass
        elif move == 'Leaf/Leaf':
            pass
        else:
            raise KeyError(f'Unrecognized Move Type: {move}')
            # case 'Node/Leaf':
            #     stats['num_options'] = self.get_num_options()
            # case 'Leaf/Node':
            #     stats['num_options'] = self.get_num_options()
        return stats

    # currently assumes operators of order 1 or 2 only
    def get_num_root_options(self):
        num_leaves = len(self._variables) + len(self.get_parameters())
        num_rr = len(self._filter_primitives(1)) + len(self._filter_primitives(2)) * num_leaves
        return num_rr

    # def get_num_options(self):
    #     num_options = sum(
    #         [
    #             int(len(self.ets[oi]) > 0 and (self.size + of - oi) <= self.max_size)
    #             for oi, of in self.move_types
    #         ]
    #     )
    #     return num_options

    # p(accept) rules courtesy of Guimera 2020: https://www.science.org/doi/10.1126/sciadv.aav6971
    # elementary tree p(accept) rule changed based on believed mistakes in author's code
    @staticmethod
    def mcmc_ruling(move, **kwargs):
        mdl_change = kwargs['new_mdl'] - kwargs['old_mdl']
        if move == 'None/Root':
            accept = add_root(mdl_change=mdl_change, num_rr=kwargs['num_rr'])
        elif move == 'Root/None':
            accept = remove_root(mdl_change=mdl_change)
        elif move == 'Node/Node':
            accept = replace_node(mdl_change=mdl_change)
        elif move == 'Leaf/Leaf':
            accept = replace_node(mdl_change=mdl_change)
        elif move == 'Node/Leaf':
            # accept = replace_elementary_tree(mdl_change=mdl_change,
            #                                  nfi=kwargs['num_options'], nif=self.get_num_options(),
            #                                  )
            accept = replace_node(mdl_change=mdl_change)
        elif move == 'Leaf/Node':
            # TODO: determine how to solve elementary tree p(accept) - very involved to implement
            # accept = replace_elementary_tree(mdl_change=mdl_change,
            #                                  nfi=kwargs['num_options'], nif=self.get_num_options(),
            #                                  )
            accept = replace_node(mdl_change=mdl_change)
        else:
            raise KeyError(f'Unrecognized move type: {move}')
        return accept
