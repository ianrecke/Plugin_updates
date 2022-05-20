"""
Pearson correlation structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np

from .learn_structure import LearnStructure
from ..util.data_structures import matrix2nx


class Pearson(LearnStructure):
    """
    Pearson correlation structure learning class.
    """

    def run(self, env='neurogenpy'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='neurogenpy'
            Environment used to run the algorithm. Currently supported:

                - 'neurogenpy'

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

        if env == 'neurogenpy':
            return self._run_neurogenpy()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):
        nodes_names = list(self.data.columns.values)
        corr_matrix = self.data.corr()

        adj_matrix = np.array(corr_matrix)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = np.square(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes_names)

        return graph
