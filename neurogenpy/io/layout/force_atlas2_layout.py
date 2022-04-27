import networkx
from fa2 import ForceAtlas2

from .graph_layout import GraphLayout


class ForceAtlas2Layout(GraphLayout):
    """
    ForceAtlas2 layout class.
    """

    def run(self, env='fa2'):
        """
        Calculates the layout for the graph with the ForceAtlas2 algorithm.

        Parameters
        ----------
        env : str, default='env2'
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

        if env == 'fa2':
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
                gravity=25.01,

                # Log
                verbose=True
            )

            initial_layout = networkx.circular_layout(self.graph)

            return forceatlas2.forceatlas2_networkx_layout(self.graph,
                                                           pos=initial_layout,
                                                           iterations=100)
        else:
            raise ValueError(f'{env} environment is not supported.')