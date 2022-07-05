"""
Structure learning base module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from abc import abstractmethod, ABCMeta

from rpy2.robjects.packages import importr

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

    data_type : {'continuous', 'discrete'}
        Type of the data introduced.

    """

    def __init__(self, df, data_type):
        self.df = df
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
        """

        dataframe = pd2r(self.df)
        nodes = list(self.df.columns.values)

        output_raw_r = importr('bnlearn').cextend(
            bnlearn_function(dataframe, **kwargs))

        graph = bnlearn2nx(nodes, output_raw_r)

        return graph
