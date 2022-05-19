"""
Structure learning base module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from abc import abstractmethod, ABCMeta

from ..util.data_structures import pd2r, bnlearn2nx


# TODO: Inherit parameters part of docstring in those subclasses that use
#  superclass constructor.
class LearnStructure(metaclass=ABCMeta):
    """
    Base class for all structure learning classes.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    """

    def __init__(self, df, data_type):
        self.data = df
        self.data_type = data_type

    @abstractmethod
    def run(self, env=None):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, optional
            Environment used to run the algorithm.

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.
        """

    # TODO: Add bnlearn whitelist and blacklist
    def _run_bnlearn(self, bnlearn_function, **kwargs):
        """
        Run a structure learning algorithm using its implementation in R's
        bnlearn.

        Parameters
        ----------
        bnlearn_function :
            R's bnlearn structure learning function to execute.

        kwargs:
            Additional parameters for the learning function.


        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the `data_type` is not supported.
        """

        if self.data_type == 'hybrid':
            raise ValueError(
                'This algorithm does not support hybrid Bayesian networks')
        dataframe = pd2r(self.data)

        nodes = list(self.data.columns.values)

        output_raw_r = bnlearn_function(dataframe, **kwargs)

        graph = bnlearn2nx(nodes, output_raw_r)

        return graph
