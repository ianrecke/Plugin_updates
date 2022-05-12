"""
igraph layout module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:


<<<<<<< HEAD
from .graph_layout import GraphLayout
from ...utils.data_structures import nx2igraph


class IgraphLayout(GraphLayout):
=======
from .layout import Layout
from ...utils.data_structures import nx2igraph


class IgraphLayout(Layout):
>>>>>>> dev
    """
    Class for igraph layouts. Any layout provided by igraph can be used.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph whose layout has to be computed.

    layout_name : str
        The igraph layout name of the layout to use.
    """

    def __init__(self, graph, *, layout_name):
        super().__init__(graph)
        self.layout_name = layout_name

<<<<<<< HEAD
    def run(self, env='igraph'):
=======
    def run(self, env='igraph', bbox=(60, 60), **_):
>>>>>>> dev
        """
        Calculates the layout for the graph.

        Parameters
        ----------
        env : str
            Environment used to calculate the layout.

<<<<<<< HEAD
=======
        bbox : (int, int), default=(600, 600)
            Bounding box for the graph.

>>>>>>> dev
        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.
        """

        graph = nx2igraph(self.graph)
        nodes = list(self.graph.nodes())

        layout = graph.layout(self.layout_name)
<<<<<<< HEAD
=======
        layout.fit_into(bbox=bbox)
>>>>>>> dev
        return {node: (layout[i][0], layout[i][1]) for i, node in
                enumerate(nodes)}
