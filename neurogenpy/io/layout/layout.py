"""
Graph layout module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from abc import ABCMeta, abstractmethod


class Layout(metaclass=ABCMeta):
    """
    Base class for all layout classes.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph whose layout has to be computed.
    """

    def __init__(self, graph):
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
