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

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    Raises
    ------
    ValueError
        If data is not continuous.
    """

    def __init__(self, df, data_type):
        if data_type != 'continuous':
            raise Exception(
                'Algorithm only supported for continuous datasets.')
        super().__init__(df, data_type)

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

        n, g = data.shape
        adj_matrix = np.zeros((g, g))
        for i in range(g):
            x = data[:, np.array([j != i for j in range(g)])]
            y = data[:, i]
            regression = LinearRegression().fit(x, y)
            adj_matrix[i] = np.concatenate(
                (regression.coef_[:i], [0], regression.coef_[i:]))

        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes)

        return graph
