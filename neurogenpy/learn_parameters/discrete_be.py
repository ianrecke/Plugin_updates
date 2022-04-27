from pgmpy.estimators import BayesianEstimator

from .learn_parameters import LearnParameters
from ..utils.data_structures import nx2pgmpy


class DiscreteBE(LearnParameters):
    """
    Discrete Bayesian parameter estimation.

    Parameters
    ----------
    data : pandas DataFrame
        Input data used to learn the parameters from.

    data_type : {'continuous', 'discrete', 'hybrid'}
        Type of the data introduced.

    graph : networkx.DiGraph
        Structure of the Bayesian network.
    """

    def __init__(self, data, data_type, graph, *, prior='BDeu',
                 equivalent_size=5):
        super().__init__(data, data_type, graph)
        self.prior = prior
        if self.prior == 'BDeu':
            self.eq_size = equivalent_size

    def run(self, env='pgmpy'):
        """
        Learns the parameters of the network using Discrete Bayesian
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
