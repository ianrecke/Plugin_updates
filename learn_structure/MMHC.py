from rpy2.robjects.packages import importr

from .LearnStructure import LearnStructure


class MMHC(LearnStructure):
    """MMHC structure learning class."""

    def __init__(self, data, data_type, states_names=None):
        """
        MMHC structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(MMHC, self).__init__(data, data_type, states_names)

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        model = None
        if backend == "neurosuites":
            model = self.run_mmhc_neurosuites()
        elif backend == "bnlearn":
            model = self.run_bnlearn(importr("bnlearn").mmhc)

        return model

    def run_mmhc_neurosuites(self):

        return 0
