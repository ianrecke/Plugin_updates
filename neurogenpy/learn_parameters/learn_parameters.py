from abc import ABCMeta, abstractmethod


class LearnParameters(metaclass=ABCMeta):
    """
    Base class for all learn parameters classes.

    Parameters
    ----------
    data : pandas DataFrame
        Input data used to learn the parameters from.

    data_type : {'continuous', 'discrete', 'hybrid'}
        Type of the data introduced.

    graph : networkx.DiGraph
        Structure of the Bayesian network.
    """

    def __init__(self, data, data_type, graph):
        self.data = data
        self.data_type = data_type
        self.graph = graph

    @abstractmethod
    def run(self):
        """
        Learns the parameters of the Bayesian network.
        """
