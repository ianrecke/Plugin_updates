from rpy2.robjects.packages import importr
import numpy as np

from .LearnStructure import LearnStructure, parse_output_structure_bnlearn, \
    pd2r


class Tan(LearnStructure):
    """Tree augmented naive Bayes structure learning class."""

    def __init__(self, data, data_type, features_classes, states_names=None):
        """
        Tree augmented naive Bayes structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param features_classes:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(Tan, self).__init__(data, data_type, states_names)
        self.features_classes = features_classes
        if len(self.features_classes) == 0:
            raise Exception(
                "To run this classifier, you must supply one class feature in the previous section.")

    def run(self, backend="bnlearn"):
        """

        @param backend:
        @return:
        """
        nodes = list(self.data.columns.values)

        model = None
        if backend == "neurosuites":
            model = self.run_tan_neurosuites(nodes)
        elif backend == "bnlearn":
            model = self.run_tan_bnlearn(nodes)

        return model

    def run_tan_bnlearn(self, nodes):
        """

        @param nodes:
        @return:
        """
        dataframe = pd2r(self.data, self.data_type)

        bnlearn = importr("bnlearn")
        try:
            nodes.remove(self.features_classes[0])
        except Exception as e:
            pass  # Class feature is not in the set of predictor features so no need to remove it from the predictor features set
        nodes = np.array(nodes)
        output_raw_r = bnlearn.tree_bayes(
            x=dataframe,
            training=self.features_classes[0],
            explanatory=nodes)

        graph = parse_output_structure_bnlearn(nodes, output_raw_r)

        return graph

    def run_tan_neurosuites(self, nodes):

        return 0
