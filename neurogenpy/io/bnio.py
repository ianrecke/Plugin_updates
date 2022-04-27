from abc import ABCMeta, abstractmethod


class BNIO(metaclass=ABCMeta):
    """
    Base class for all input/output Bayesian network classes.
    """

    def __init__(self):
        pass

    @abstractmethod
    def read_file(self, file_path):
        """
        Reads a Bayesian network from a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored.

        Returns
        -------
        networkx.DiGraph
            The graph structure of the loaded Bayesian network.
        """

    @abstractmethod
    def write_file(self, file_path, bn):
        """
        Writes a Bayesian network in a file.

        Parameters
        ----------
        file_path :
            Path of the file to store the Bayesian network in.

        bn : BayesianNetwork
            Bayesian network to be stored.
        """
