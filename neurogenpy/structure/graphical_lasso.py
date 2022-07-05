"""
Graphical lasso structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np
from sklearn import covariance as sk_learn_cov

from .learn_structure import LearnStructure
from ..util.data_structures import matrix2nx


class GraphicalLasso(LearnStructure):
    """
    Graphical lasso structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    alpha : float, default=0.5
        The regularization parameter. See sklearn documentation for more
        information.

    tol : float, default=1e-4
        The tolerance to declare convergence. See sklearn documentation for
        more information.

    max_iter : int, default=100
        The maximum number of iterations.

    Raises
    ------
    ValueError
        If the data is not continuous.
    """

    def __init__(self, df, data_type, *, alpha=0.5, tol=1e-4, max_iter=100):
        if data_type != 'continuous':
            raise Exception(
                'Algorithm only supported for continuous datasets.')

        super().__init__(df, data_type)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def run(self, env='scikit-learn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='scikit-learn'
            Environment used to run the algorithm. Currently supported:

                - 'scikit-learn': :cite:`glasso`

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == 'scikit-learn':
            return self._run_sklearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_sklearn(self):
        nodes = list(self.df.columns.values)
        data = np.array(self.df)

        cov_matrix = np.cov(data.T)
        _, precision_matrix = sk_learn_cov.graphical_lasso(
            cov_matrix, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter)
        adj_matrix = np.array(precision_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes)

        return graph
