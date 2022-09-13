"""
Grow shrink structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class GrowShrink(LearnStructure):
    """
    Grow shrink structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    alpha: float, default=0.5
        The target nominal type I error rate. See bnlearn documentation for
        more information.
    """

    def __init__(self, df, data_type=None, *, alpha=0.05):

        super().__init__(df, data_type)
        self.alpha = alpha

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`constraint`

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """
        if env == "neurogenpy":
            return self._run_neurogenpy()
        elif env == "bnlearn":
            kwargs = {'alpha': self.alpha}
            if self.data_type == 'discrete':
                kwargs['test'] = 'mi'
            return self._run_bnlearn(importr("bnlearn").gs, **kwargs)
        else:
            raise ValueError(f"{env} environment is not supported.")

    def _run_neurogenpy(self):

        return None
