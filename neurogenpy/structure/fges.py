"""
FGES structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from ._fges_base import FGESBase, FGESStructure
from ..util.data_structures import matrix2nx


class FGES(FGESBase):
    """
    FGES algorithm class.
    It follows :cite:`fges`, but using the Bayesian Information Criteria rather
    than mutual information as described in :cite:`fges_merge`.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous'}
        Type of the data introduced.

    penalty : int, default=45
        Penalty hyperparameter of the FGES algorithm.
    """

    def __init__(self, df, data_type, *, penalty=45):
        if self.data_type != 'continuous':
            raise ValueError(
                'This algorithm is only available for continuous data.')

        super().__init__(df, data_type, penalty=penalty)

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
        """
        if env == 'neurogenpy':
            return self._run_neurogenpy()

        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):
        self._setup()

        fges_structure = FGESStructure(self.data, self.bics, self.penalty,
                                       self.n_jobs)
        self.graph = fges_structure.run()
        return matrix2nx(self.graph, self.nodes_ids)
