import numpy as np

from .LearnStructure import LearnStructure, nx_graph_from_adj_matrix


class Pearson(LearnStructure):
    """Pearson correlation structure learning class."""

    def __init__(self, data, data_type, states_names=None):
        """
        Pearson correlation structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(Pearson, self).__init__(data, data_type, states_names)

    def run(self, backend="neurosuites"):
        """

        @param backend:
        @return:
        """
        if self.data_type != "continuous":
            raise Exception(
                "Algorithm only supported for continuous datasets ")

        nodes_names = list(self.data.columns.values)

        if backend == "neurosuites":
            graph = self.run_pearson_neurosuites(nodes_names)
        else:
            raise Exception("Backend {} is not supported.".format(backend))

        return graph

    def run_pearson_neurosuites(self, nodes_names):
        """

        @param nodes_names:
        @return:
        """
        corr_matrix = self.data.corr()

        adj_matrix = np.array(corr_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = nx_graph_from_adj_matrix(adj_matrix, nodes_names)

        return graph
