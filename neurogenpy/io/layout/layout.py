"""
Graph layout module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from abc import ABCMeta, abstractmethod


class Layout(metaclass=ABCMeta):
    """
    Base class for all layout classes.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph whose layout has to be computed.
    """

    def __init__(self, graph, **_):
        self.graph = graph

    @abstractmethod
    def run(self, env):
        """
        Calculates the layout for the graph.

        Parameters
        ----------
        env : str
            Environment used to calculate the layout.

        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.
        """
