"""
PC structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from pgmpy.estimators.PC import PC as PGMPY_PC
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure
from ..util.data_structures import pgmpy2nx


class PC(LearnStructure):
    """
    PC structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    alpha : float, default=0.5
        The target nominal type I error rate. See bnlearn documentation for
        more information.
    """

    def __init__(self, df, data_type, *, alpha=0.05):
        super().__init__(df, data_type)
        self.alpha = alpha

    def run(self, env="bnlearn"):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`constraint`
                - 'pgmpy': :cite:`pgmpy_pc`

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == "pgmpy":
            graph, _ = pgmpy2nx(
                PGMPY_PC(self.data).estimate(significance_level=self.alpha))
            return graph
        elif env == "bnlearn":
            return self._run_bnlearn(importr("bnlearn").pc_stable,
                                     alpha=self.alpha)
        else:
            raise ValueError(f"{env} environment is not supported.")
