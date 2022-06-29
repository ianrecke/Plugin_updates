"""
Discrete maximum likelihood estimation module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from pgmpy.estimators import MaximumLikelihoodEstimator

from .learn_parameters import LearnParameters
from ..util.data_structures import nx2pgmpy


class DiscreteMLE(LearnParameters):
    """
    Discrete maximum likelihood estimation.
    """

    def run(self, env='pgmpy'):
        """
        Learns the parameters of the network using Discrete maximum likelihood
        estimation.

        Parameters
        ----------
        env : {'pgmpy2'}, default='pgmpy2'
            Environment used to learn the parameters.

        Returns
        -------
        list
            List of TabularCPDs.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == 'pgmpy':
            return self._run_pgmpy()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_pgmpy(self):
        pgmpy_model = nx2pgmpy(graph=self.graph, parameters={})

        mle = MaximumLikelihoodEstimator(pgmpy_model, self.data)
        cpds = mle.get_parameters()
        nodes = list(self.data.columns.values)

        return {node: cpds[i] for i, node in enumerate(nodes)}
