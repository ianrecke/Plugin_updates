from pgmpy import estimators as pgmpy_estimators

from .LearnParameters import LearnParameters
from ..io.IOBN import nx_graph_parameters_to_pgmpy_model


class DiscreteBE(LearnParameters):
    def __init__(self, data, data_type, graph, algorithm_parameters):
        """

        @param data:
        @param data_type:
        @param graph:
        @param algorithm_parameters:
        """
        super(DiscreteBE, self).__init__(data, data_type,
                                         graph)
        self.prior = algorithm_parameters["bayesianEstimation_prior"]
        if self.prior == "BDeu":
            self.equivalent_size = algorithm_parameters[
                "bayesianEstimation_equivalent_size"]

    def run(self, backend="bnlearn"):
        model = None
        if backend == "pgmpy2":
            model = self.run_bayesian_estimation_pgmpy()

        return model

    def run_bayesian_estimation_pgmpy(self):
        bn_model_pgmpy = nx_graph_parameters_to_pgmpy_model(
            graph=self.graph, parameters={})

        bayesian_estimator_pgmpy = pgmpy_estimators.BayesianEstimator(
            bn_model_pgmpy, self.data)
        if self.prior == "K2":
            model_parameters = bayesian_estimator_pgmpy.get_parameters(
                prior_type='K2')
        elif self.prior == "BDeu":
            model_parameters = bayesian_estimator_pgmpy.get_parameters(
                prior_type='BDeu',
                equivalent_sample_size=self.equivalent_size)

        return model_parameters
