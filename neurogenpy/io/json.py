"""
Adjacency matrix input/output module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:
import json

import networkx
from networkx.readwrite import json_graph

from .bnio import BNIO


class JSON(BNIO):
    """
    Adjacency matrix input/output class.
    """

    def read_file(self, file_path):
        """
        Returns the graph structure of a Bayesian network stored in a JSON
        file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored. It can be a CSV or
            parquet file.

        Returns
        -------
        networkx.DiGraph
            The graph structure of the loaded Bayesian network.
        """

        pass

    def write_file(self, file_path, bn):
        """
        Writes a Bayesian network structure in JSON format.

        Parameters
        ----------
        file_path: str
            Path of the file to store the Bayesian network in. It can be a CSV
            or parquet file.

        bn : BayesianNetwork
            Bayesian network to be stored.
        """

        pass

    def generate(self, bn, options=None, keys=None):
        """

        Parameters
        ----------
        bn

        options

        keys

        Returns
        -------

        """
        data = json_graph.node_link_data(bn.graph, options)
        if keys:
            data = {k: data[k] for k in keys}
        return json.dumps(data)

    def convert(self, io_object):
        pass
