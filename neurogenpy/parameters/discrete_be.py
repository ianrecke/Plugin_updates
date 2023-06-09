"""
Discrete Bayesian estimation module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from pgmpy.estimators import BayesianEstimator

from .learn_parameters import LearnParameters
from ..util.data_structures import nx2pgmpy


class DiscreteBE(LearnParameters):
    """
    Discrete Bayesian parameter estimation.

    Parameters
    ----------
    data : pandas DataFrame
        Input data used to learn the parameters from.

    graph : networkx.DiGraph
        Structure of the Bayesian network.
    """

    def __init__(self, data, *, graph=None, prior='BDeu', equivalent_size=5):
        super().__init__(data, graph)
        self.prior = prior
        if self.prior == 'BDeu':
            self.eq_size = equivalent_size

    def run(self, env='pgmpy'):
        """
        Learns the parameters of the network using Discrete Bayesian
        estimation.

        Parameters
        ----------
        env : {'pgmpy'}, default='pgmpy'
            Environment used to learn the parameters.

        Returns
        -------
        dict
            Nodes as keys and TabularCPDs as values.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == 'pgmpy':
            return self._run_pgmpy()
        else:
            raise ValueError(f'Backend {env} is not supported.')

    def _run_pgmpy(self):
        pgmpy_model = nx2pgmpy(graph=self.graph, parameters={})
        nodes = list(self.data.columns.values)

        be = BayesianEstimator(pgmpy_model, self.data)
        if self.prior == 'K2':
            cpds = be.get_parameters(prior_type='K2')
        elif self.prior == 'BDeu':
            cpds = be.get_parameters(prior_type='BDeu',
                                     equivalent_sample_size=self.eq_size)
        else:
            raise ValueError(
                f'Prior distribution {self.prior} is not supported.')

        return {node: cpds[i] for i, node in enumerate(nodes)}
