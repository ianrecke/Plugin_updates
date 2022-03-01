from sklearn import covariance as sk_learn_cov
import numpy as np

from .LearnStructure import LearnStructure, nx_graph_from_adj_matrix


class Glasso(LearnStructure):
    """Graphical lasso structure learning class."""

    def __init__(self, data, data_type, alpha, tol, max_iter,
                 states_names=None):
        """
        Graphical lasso structure learning constructor.
        @param data: DataFrame with the learning sample from which to infer the
            network.
        @type data: Pandas DataFrame.
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param alpha:
        @param tol:
        @param max_iter:
        @param states_names: Dictionary with the set of states each variable
            takes in the input data.
        """
        super(Glasso, self).__init__(data, data_type, states_names)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

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
            graph = self.run_glasso_scikit_learn(nodes_names)
        else:
            raise Exception("Backend {} is not supported.".format(backend))

        return graph

    def run_glasso_scikit_learn(self, nodes_names):
        """

        @param nodes_names:
        @return:
        """
        data_np = np.array(self.data)

        cov_matrix = np.cov(data_np.T)
        _, precision_matrix = sk_learn_cov.graphical_lasso(
            cov_matrix, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter)
        adj_matrix = np.array(precision_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = nx_graph_from_adj_matrix(adj_matrix, nodes_names)

        return graph
