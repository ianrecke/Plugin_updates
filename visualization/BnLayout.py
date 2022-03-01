from abc import ABCMeta, abstractmethod
import igraph


class BnLayout(metaclass=ABCMeta):
    """
    Base class for all Layout classes.
    """

    def __init__(self, graph, graph_initial_width=None,
                 graph_initial_height=None):
        """
        Layout class constructor.
        @param graph:
        @param graph_initial_width:
        @param graph_initial_height:
        """
        self.graph = graph
        self.layout = {}
        self.graph_initial_width = graph_initial_width
        self.graph_initial_height = graph_initial_height

    @abstractmethod
    def run(self):
        """

        """
        pass


# TODO: move networkx_to_igraph to utils/helpers
def networkx_to_igraph(graph_networkx):
    graph_igraph = igraph.Graph(directed=True)
    nodes = list(graph_networkx.nodes())
    edges = list(graph_networkx.edges())
    graph_igraph.add_vertices(nodes)
    graph_igraph.add_edges(edges)

    return graph_igraph
