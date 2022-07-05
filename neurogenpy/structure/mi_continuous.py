"""
Mutual information structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .learn_structure import LearnStructure
from ..util.data_structures import matrix2nx


class MiContinuous(LearnStructure):
    """
    Mutual information structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    Raises
    ------
    ValueError
        If the data is not discrete
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

                - 'scikit-learn': :cite:`mir`

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
        mi_matrix = []
        data = np.array(self.df)

        for i in range(self.df.shape[1]):
            y = data[:, i]
            mi_y = mutual_info_regression(data, y, discrete_features=False)
            mi_matrix.append(mi_y)

        mi_matrix = np.array(mi_matrix)
        adj_matrix = np.array(mi_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes)

        return graph
