"""
igraph layout module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html


from .layout import Layout
from ...util.data_structures import nx2igraph


class IgraphLayout(Layout):
    """
    Class for igraph layouts. Any layout provided by igraph can be used.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph whose layout has to be computed.

    layout_name : str
        The igraph layout name of the layout to use. 2D igraph available
        layouts are:

            - Automatic layout ('auto', 'automatic')
            - Bipartite layout ('bipartite')
            - Circular layout ('circle', 'circular')
            - Davidson-Harel layout ('dh', 'davidson_harel')
            - DrL layout for large graphs ('drl')
            - Fruchterman-Reingold layout ('fr', 'fruchterman_reingold')
            - Regular grid layout in 2D ('grid')
            - The graphopt algorithm ('graphopt')
            - Kamada-Kawai layout ('kk', 'kamada_kawai')
            - Large Graph Layout ('lgl', 'large', 'large_graph')
            - Multidimensional scaling layout ('mds')
            - Random layout ('random')
            - Reingold-Tilford tree layout ('rt', 'tree', 'reingold_tilford')
            - Circular Reingold-Tilford tree layout ('rt_circular',
                'reingold_tilford_circular')
            - Star layout ('star')
            - Sugiyama layout ('sugiyama')
    """

    def __init__(self, graph, *, layout_name):
        super().__init__(graph)
        self.layout_name = layout_name

    def run(self, env='igraph', bbox=(60, 60)):
        """
        Calculates the layout for the graph.

        Parameters
        ----------
        env : str
            Environment used to calculate the layout.

        bbox : (int, int), default=(600, 600)
            Bounding box for the graph.

        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.
        """

        graph = nx2igraph(self.graph)
        nodes = list(self.graph.nodes())

        layout = graph.layout(self.layout_name)
        layout.fit_into(bbox=bbox)
        return {node: (layout[i][0], layout[i][1]) for i, node in
                enumerate(nodes)}
