from rpy2.robjects.packages import importr

from .LearnStructure import LearnStructure


class HcTabu(LearnStructure):
    """Hill climbing with tabu search structure learning class."""

    def __init__(self, data, max_number_parents, iterations, data_type=None,
                 states_names=None):
        """
        Hill climbing with tabu search structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param max_number_parents:
        @param iterations:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(HcTabu, self).__init__(data, data_type, states_names)
        self.max_number_parents = max_number_parents
        self.iterations = iterations

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        model = None
        if backend == "neurosuites":
            model = self.run_hc_tabu_neurosuites()
        elif backend == "bnlearn":
            model = self.run_bnlearn(importr("bnlearn").tabu)

        return model

    def run_hc_tabu_neurosuites(self):

        return 0
