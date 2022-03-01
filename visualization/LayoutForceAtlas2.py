from .BnLayout import BnLayout
from fa2 import ForceAtlas2
import networkx


class LayoutForceAtlas2(BnLayout):
    """
    ForceAtlas2 Layout class.
    """

    def __init__(self, graph, graph_initial_width=None,
                 graph_initial_height=None):
        """
        ForceAtlas2 Layout class constructor.
        @param graph:
        @param graph_initial_width:
        @param graph_initial_height:
        """
        super(LayoutForceAtlas2, self).__init__(graph, graph_initial_width,
                                                graph_initial_height)

    def run(self):
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=0.0,

            # Performance
            jitterTolerance=10.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=2.01,
            strongGravityMode=False,
            gravity=25.01,  # 25.01,

            # Log
            verbose=True
        )

        """
        initial_layout = {}
        for i, node in enumerate(graph_networkx.nodes()):
            pos = np.array([1, i])
            initial_layout[node] = pos
        """
        initial_layout = networkx.circular_layout(self.graph)

        self.layout = forceatlas2.forceatlas2_networkx_layout(
            self.graph, pos=initial_layout, iterations=100)

        return self.layout
