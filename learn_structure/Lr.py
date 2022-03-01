import numpy as np
from sklearn.linear_model import LinearRegression

from .LearnStructure import LearnStructure, nx_graph_from_adj_matrix


class Lr(LearnStructure):
    """Linear regression structure learning class."""

    def __init__(self, data, data_type, states_names=None):
        """
        Linear regression structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(Lr, self).__init__(data, data_type, states_names)

    def run(self, backend="scikit-learn"):
        """

        @param backend:
        @return:
        """
        if self.data_type != "continuous":
            raise Exception(
                "Algorithm only supported for continuous datasets ")

        nodes_names = list(self.data.columns.values)

        if backend == "scikit-learn":
            graph = self.run_lr_scikit_learn(nodes_names)
        else:
            raise Exception("Backend {} is not supported.".format(backend))

        return graph

    def run_lr_scikit_learn(self, nodes_names):
        """

        @param nodes_names:
        @return:
        """
        data_np = np.array(self.data)

        n, g = data_np.shape
        adj_matrix = np.zeros((g, g))
        for i in range(g):
            x = data_np[:, np.array([j != i for j in range(g)])]
            y = data_np[:, i]
            regression = LinearRegression().fit(x, y)
            adj_matrix[i] = np.concatenate(
                (regression.coef_[
                 :i], [0], regression.coef_[
                           i:]))

        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = nx_graph_from_adj_matrix(adj_matrix, nodes_names)

        return graph
