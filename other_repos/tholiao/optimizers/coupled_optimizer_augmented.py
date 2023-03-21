import numpy as np
import time

from DIRECT import solve
from scipy.optimize import minimize

from .coupled_optimizer import JointBayesOptimizer


class JointOptimizerAug(JointBayesOptimizer):
    def __init__(self, obj_f, n_uc, init_uc, bounds_uc, uc_runs_per_cn, init_cn,
                 bounds_cn, n_cn, contextual=True, uc_to_return='max',
                 start_with_x=None, start_with_y=None, no=None):
        if no is None:
            raise ValueError("no can't be None.")
        super(JointOptimizerAug, self).__init__(obj_f, n_uc, init_uc, bounds_uc,
                                                uc_runs_per_cn, init_cn,
                                                bounds_cn, n_cn,
                                                contextual=contextual,
                                                uc_to_return=uc_to_return,
                                                start_with_x=start_with_x,
                                                start_with_y=start_with_y, no=no)


    def initialize_GP(self, n_init, x_cn):
        self.update_iterations(i=self.init_uc)

        x_uc = self.random_parameters(self.init_uc, self.no.get_seed())
        x_cn = np.tile(x_cn, (self.init_uc, 1))

        self.X = np.concatenate((x_uc, x_cn), axis=1)
        self.Y = self.evaluate(self.X)
        self.Y_mean = np.zeros((self.X.shape[0], 1))
        self.Y_var = np.zeros((self.X.shape[0], 1))
        self.train_GP(self.X, self.Y)

        self.optimize_model()

        print("Done initializing GP_uc")

    def eval_hw(self, x_cn, cache_walker=True, test=False):
        """
        Used as objective function by hw_optimizer.
        Given a context, optimize x_sw and return the reward from obj_f
        :param x_cn: Used as a context during optimization
        :return: Reward from obj_f
        """
        print(f"Evaluating morphology {x_cn}")
        print(self.no)

        total_evaluations = self.no.get_inner_quantity()
        initialize_evaluations = min(self.init_uc, total_evaluations)

        self.initialize_GP(initialize_evaluations, x_cn)

        print("SW - optimizing with {} as context".format(x_cn))

        for i in range(total_evaluations - initialize_evaluations):
            if not test:
                self.update_iterations()
            x_uc = self.optimize_acq_f(x_cn)
            X = np.concatenate((x_uc, x_cn), axis=1)
            Y = self.evaluate(X)
            self.update_X(X)
            self.update_Y(Y)

            self.train_GP(self.X, self.Y)
            self.optimize_model()

        objective_value_x_cn = self.select_y_to_return()
        print(x_cn, float(objective_value_x_cn))
        if not test:
            self.no.next_outer(float(objective_value_x_cn), -1, -1, -1)



        if self.no.is_reevaluating_flag and not test:
            reeval_obj = float(self.eval_hw(x_cn, cache_walker, True))
            self.no.next_reeval(reeval_obj, -1, -1, -1)


        return objective_value_x_cn
