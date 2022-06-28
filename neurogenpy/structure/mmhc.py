"""
MMHC structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class MMHC(LearnStructure):
    """
    MMHC structure learning class.
    """

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`hybrid`

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
            return self._run_bnlearn(importr('bnlearn').mmhc)
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):

        return None
