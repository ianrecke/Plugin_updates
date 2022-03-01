import numpy as np
from sklearn.feature_selection import mutual_info_regression as sklearn_mi

from .LearnStructure import LearnStructure, nx_graph_from_adj_matrix


class MiContinuous(LearnStructure):
    """Mutual information structure learning class."""

    def __init__(self, data, data_type, states_names=None):
        """
        Mutual information structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(MiContinuous, self).__init__(data, data_type, states_names)

    def run(self, backend="neurosuites"):
        """

        @param backend:
        @return:
        """
        if self.data_type != "continuous":
            raise Exception(
                "Algorithm only supported for continuous datasets ")

        nodes_names = list(self.data.columns.values)

        if backend == "scikit-learn":
            graph = self.run_mi_continuous_scikit_learn(nodes_names)
        else:
            raise Exception("Backend {} is not supported.".format(backend))

        return graph

    def run_mi_continuous_scikit_learn(self, nodes_names):
        """

        @param nodes_names:
        @return:
        """
        mi_matrix = []
        data_np = np.array(self.data)

        for i in range(self.data.shape[1]):
            y = data_np[:, i]
            mi_y = sklearn_mi(data_np, y)
            mi_matrix.append(mi_y)

        mi_matrix = np.array(mi_matrix)
        adj_matrix = np.array(mi_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)

        graph = nx_graph_from_adj_matrix(adj_matrix, nodes_names)

        return graph
