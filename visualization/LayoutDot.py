import networkx

from .BnLayout import BnLayout


class LayoutDot(BnLayout):
    """
    Dot Layout class.
    """

    def __init__(self, graph, graph_initial_width=None,
                 graph_initial_height=None):
        """

        @param graph:
        @param graph_initial_width:
        @param graph_initial_height:
        """
        super(LayoutDot, self).__init__(graph, graph_initial_width,
                                        graph_initial_height)

    def run(self):
        layout_reversed = networkx.drawing.nx_agraph.graphviz_layout(
            self.graph, prog='dot')

        for node_reversed in layout_reversed:
            self.layout[node_reversed] = (
                (layout_reversed[node_reversed][0]) / self.graph_initial_width,
                (layout_reversed[node_reversed][
                     1] * -1) / self.graph_initial_height
            )

        return self.layout
