"""
Dot layout module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import networkx as nx

from .layout import Layout


class DotLayout(Layout):
    """
    Dot layout class.
    """

    def run(self, env='networkx', **_):
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

        if env == 'networkx':
            layout = nx.drawing.nx_agraph.graphviz_layout(self.graph,
                                                          prog='dot')
            return {k: (v[0], -v[1]) for k, v in layout.items()}
        else:
            raise ValueError(f'{env} environment is not supported.')
