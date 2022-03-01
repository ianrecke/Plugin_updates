from rpy2.robjects.packages import importr
import numpy as np
import pandas as pd

from .LearnStructure import LearnStructure, parse_output_structure_bnlearn, \
    pd2r


class MBC(LearnStructure):
    """Multi-dimensional Bayesian network classifier class."""

    def __init__(self, data, data_type, features_classes,
                 states_names=None):
        """
        Multi-dimensional Bayesian network classifier constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param features_classes:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(MBC, self).__init__(data, data_type, states_names)
        self.features_classes = features_classes
        if len(self.features_classes) == 0:
            raise Exception(
                "To run this classifier, you must supply at least one class feature in the previous section.")

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        nodes = list(self.data.columns.values)

        model = None
        if backend == "neurosuites":
            model = self.run_mbc_neurosuites(nodes)
        elif backend == "bnlearn":
            model = self.run_mbc_bnlearn(nodes)

        return model

    def run_mbc_bnlearn(self, nodes):
        """

        @param nodes:
        @return:
        """
        dataframe = pd2r(self.data, self.data_type)

        bnlearn = importr("bnlearn")
        features = list(set(nodes) - set(self.features_classes))

        # Black list of arcs from features to classes
        # Shape = (len(self.features_classes) * len(features), 2)
        blacklist = pd.DataFrame(columns=["from", "to"])
        blacklist["from"] = features * len(self.features_classes)
        blacklist["to"] = np.repeat(
            self.features_classes, [
                len(features)], axis=0)
        """
        bl < - matrix(nrow=length(classes) * length(features), ncol=2, dimnames=list(NULL, c("from", "to")))
        bl[, "from"] < - rep(features, each=length(classes))
        bl[, "to"] < - rep(classes, length(features))
        """
        # Learn MBC structure
        output_raw_r = bnlearn.hc(x=dataframe, blacklist=blacklist)

        graph = parse_output_structure_bnlearn(nodes, output_raw_r)

        return graph

    def run_mbc_neurosuites(self, nodes):

        return 0
