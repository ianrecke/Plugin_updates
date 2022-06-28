"""
Hill climbing structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class HillClimbing(LearnStructure):
    """
    Hill climbing structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    maxp : int, default=100
        The maximum number of parents for a node.

    max_iter : int, default=100
        The maximum number of iterations.
    """

    def __init__(self, df, data_type, *, maxp=100, max_iter=100):

        super().__init__(df, data_type)
        self.maxp = maxp
        self.max_iter = max_iter

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`hc`

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
        elif env == 'bnlearn':
            return self._run_bnlearn(importr('bnlearn').hc, maxp=self.maxp,
                                     max_iter=self.max_iter)
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):

        return None
