import numba
import networkx as nx
import numpy as np

from .LearnParameters import LearnParameters
from ..utils.graph_utils import parents


class GaussianMLE(LearnParameters):
    def __init__(self, data, data_type, graph, algorithm_parameters):
        """

        @param data:
        @param data_type:
        @param graph:
        @param algorithm_parameters:
        """
        super(GaussianMLE, self).__init__(data, data_type, graph)
        self.algorithm_parameters = algorithm_parameters
        self.graph = nx.to_numpy_matrix(graph)

    def run(self, backend="neurosuites"):
        if backend == "neurosuites":
            model_parameters = self.run_mle_neurosuites()

        return model_parameters

    def run_mle_neurosuites(self):
        model_parameters = {}
        cols = self.data.columns.values.tolist()

        for node in cols:
            node_idx = cols.index(node)
            parents_idx = parents(self.graph, node_idx)
            parents_names = [cols[i] for i in parents_idx]

            y = self.data.loc[:, node].values.reshape(self.data.shape[0], -1)

            if len(parents_names) == 0:
                mean = y.mean()
                variance = y.var()
                parents_coeffs = []
            else:
                x = self.data.loc[:, list(parents_names)].values.reshape(
                    self.data.shape[0], -1)
                variance, coeffs = linear_gaussian(x, y)
                mean, parents_coeffs = coeffs[0], coeffs[1:]

            model_parameters[node] = GaussianNode(mean, variance,
                                                  parents_names,
                                                  parents_coeffs)

        return model_parameters


class GaussianNode:
    def __init__(self, mean, var, parents_names, parents_coeffs):
        self.mean = mean
        self.var = var
        self.parents_names = parents_names
        self.parents_coeffs = parents_coeffs

    def __str__(self):
        return "mean: {} var: {} parents_names: {} parents_coeffs {}".format(
            self.mean, self.var,
            self.parents_names, self.parents_coeffs)


@numba.jit(nopython=True, fastmath=True)
def linear_gaussian(x, y):
    n = x.shape[0]
    x = np.ascontiguousarray(x).reshape(n, -1)
    k = x.shape[1]
    y = np.ascontiguousarray(y).reshape(n, -1)

    if k == 0:
        return None

    a = np.ones((n, k + 1),
                dtype=np.float64)  # Bias is the last column of ones
    a[:, :k] = x

    result_lr = np.linalg.lstsq(a, y)
    w = result_lr[0]

    y_predicted = np.dot(w.T, a.T)
    residuals = y - y_predicted.T
    mean_sq_error = np.sum(np.square(residuals)) / n

    return mean_sq_error, list(w.flatten())
