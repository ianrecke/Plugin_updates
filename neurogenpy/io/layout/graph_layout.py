from abc import ABCMeta, abstractmethod

from ...utils.data_structures import nx2igraph


class GraphLayout(metaclass=ABCMeta):
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

    def _run_igraph(self, layout_name):
        graph = nx2igraph(self.graph)
        nodes = list(self.graph.nodes())

        layout = graph.layout(layout_name)
        return {node: (layout[i][0]) for i, node in enumerate(nodes)}
