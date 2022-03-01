from rpy2.robjects.packages import importr

from .LearnStructure import LearnStructure


class Gs(LearnStructure):
    """Grow shrink structure learning class."""

    def __init__(self, data, alpha, data_type=None,
                 states_names=None):
        """
        Growh shrink structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param alpha:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(Gs, self).__init__(data, data_type, states_names)
        self.alpha = alpha

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        if backend == "neurosuites":
            model = self.run_gs_neurosuites()
        elif backend == "bnlearn":
            model = self.run_bnlearn(importr("bnlearn").gs,
                                     self.alpha)
        return model

    def run_gs_neurosuites(self):

        return 0
