"""
Mutual information structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .learn_structure import LearnStructure
from ..utils.data_structures import matrix2nx


class MiContinuous(LearnStructure):
    """
    Mutual information structure learning class.
    """

    def run(self, env='scikit-learn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'scikit-learn'}, default='scikit-learn'
            Environment used to run the algorithm.

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        Exception
            If the data is not continuous.
        """

        if self.data_type != 'continuous':
            raise Exception(
                'Algorithm only supported for continuous datasets ')

        if env == 'scikit-learn':
            return self._run_sklearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_sklearn(self):
        """

        """

        nodes_names = list(self.data.columns.values)
        mi_matrix = []
        data_np = np.array(self.data)

        for i in range(self.data.shape[1]):
            y = data_np[:, i]
            mi_y = mutual_info_regression(data_np, y)
            mi_matrix.append(mi_y)

        mi_matrix = np.array(mi_matrix)
        adj_matrix = np.array(mi_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes_names)

        return graph
