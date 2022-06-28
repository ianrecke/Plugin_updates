"""
Linear regression structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np
from sklearn.linear_model import LinearRegression

from .learn_structure import LearnStructure
from ..util.data_structures import matrix2nx


class Lr(LearnStructure):
    """
    Linear regression structure learning class.
    """

    def run(self, env='scikit-learn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='scikit-learn'
            Environment used to run the algorithm. Currently supported:

                - 'scikit-learn': :cite:`lr`

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        Exception
            If variables are not all continuous.
        ValueError
            If the environment is not supported.
        """

        if self.data_type != 'continuous':
            raise Exception(
                'Algorithm only supported for continuous datasets ')

        if env == 'scikit-learn':
            return self._run_sklearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_sklearn(self):

        nodes_names = list(self.data.columns.values)
        data_np = np.array(self.data)

        n, g = data_np.shape
        adj_matrix = np.zeros((g, g))
        for i in range(g):
            x = data_np[:, np.array([j != i for j in range(g)])]
            y = data_np[:, i]
            regression = LinearRegression().fit(x, y)
            adj_matrix[i] = np.concatenate(
                (regression.coef_[:i], [0], regression.coef_[i:]))

        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes_names)

        return graph
