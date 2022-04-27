"""
FGES structure learning module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from ._fges_base import FGESBase, FGESStructure
from neurogenpy.utils.data_structures import matrix2nx


class FGES(FGESBase):
    """
    FGES algorithm class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete', 'hybrid'}
        Type of the data introduced.

    penalty : int, default=45
        Penalty hyperparameter of the FGES algorithm.

    References
    ----------
    .. [1] Ramsey J., Glymour M., Sanchez-Romero R., Glymour C. A million
        variables and more: The fast greedy equivalence search algorithm for
        learning high-dimensional graphical causal models, with an application
        to functional magnetic resonance images. International Journal of Data
        Science and Analytics. 2017;3:121–129.

    .. [2] N. Bernaola, M. Michiels, P. Larrañaga, C. Bielza. Learning massive
       interpretable gene regulatory networks of the human brain by merging
       Bayesian Networks, bioRxiv
       `<https://doi.org/10.1101/2020.02.05.935007>`_.
       `<https://www.biorxiv.org/content/early/2020/02/05/2020.02.05.935007>`_.
    """

    def __init__(self, df, data_type, *, penalty=45, **_):
        # TODO: Transform pandas DataFrame to numpy array?
        super().__init__(df, data_type, penalty=penalty)

    def run(self, env='neurogenpy'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'neurogenpy'}, default='neurogenpy'
            Environment used to run the algorithm.

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

        fges_structure = FGESStructure(self.data, self.bics, self.nodes,
                                       self.penalty, self.n_jobs)
        self.graph = fges_structure.run()
        return matrix2nx(self.graph, self.nodes_ids)
