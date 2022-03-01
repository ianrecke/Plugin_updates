from pgmpy import estimators as pgmpy_estimators

from .LearnParameters import LearnParameters
from ..io.IOBN import nx_graph_parameters_to_pgmpy_model


class DiscreteMLE(LearnParameters):
    def __init__(self, data, data_type, graph, algorithm_parameters):
        """

        @param data:
        @param data_type:
        @param graph:
        @param algorithm_parameters:
        """
        super(DiscreteMLE, self).__init__(data, data_type, graph)
        self.algorithm_parameters = algorithm_parameters

    def run(self, backend="bnlearn"):
        if backend == "pgmpy2":
            model_parameters = self.run_mle_pgmpy()

        return model_parameters

    def run_mle_pgmpy(self):
        bn_model_pgmpy = nx_graph_parameters_to_pgmpy_model(
            graph=self.graph, parameters={})

        mle_pgmpy = pgmpy_estimators.MaximumLikelihoodEstimator(bn_model_pgmpy,
                                                                self.data)
        model_parameters = mle_pgmpy.get_parameters()

        return model_parameters
