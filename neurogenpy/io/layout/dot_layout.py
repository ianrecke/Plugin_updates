import networkx as nx

from .graph_layout import GraphLayout


class DotLayout(GraphLayout):
    """
    Dot layout class.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph whose layout has to be computed.
    """

    def __init__(self, graph):
        super().__init__(graph)

    def run(self, env='nx'):
        """
        Calculates the layout for the graph with the dot algorithm.

        Parameters
        ----------
        env : str, default='nx'
            Environment used to calculate the layout.

        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.

        Raises
        ------
        ValueError
            If the environment selected is not supported.
        """

        if env == 'nx':
            layout = nx.drawing.nx_agraph.graphviz_layout(self.graph,
                                                          prog='dot')
            return {k: (v[0], -v[1]) for k, v in layout.items()}
        else:
            raise ValueError(f'{env} environment is not supported.')
