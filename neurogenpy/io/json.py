"""
JSON input/output module. It uses `networkx` functionality, but tries to adapt
to other similar formats such as
`Graphology <https://graphology.github.io/>`_.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import json

from networkx.readwrite import json_graph

from .bnio import BNIO


class JSON(BNIO):
    """
    JSON input/output class.
    """

    # TODO: Add options functionality
    def read_file(self, file_path):
        """
        Returns the graph structure of a Bayesian network stored in a JSON
        file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file where the network is stored.

        Returns
        -------
        BayesianNetwork
            The Bayesian Network loaded.
        """

        with open(file_path, 'r') as f:
            json_str = f.read()

            data = json.loads(json_str)
            graph = json_graph.node_link_graph(data['graph'], directed=True,
                                               multigraph=False)
            parameters = data['parameters']
            data_type = data['data_type']
            return graph, parameters, data_type

    def write_file(self, file_path):
        """
        Writes a Bayesian network structure in JSON format.

        Parameters
        ----------
        file_path: str
            Path of the JSON file to store the graph structure in.
        """

        json_str = self.generate()
        with open(file_path, 'w') as f:
            f.write(json_str)

    def generate(self, options=None, keys=None):
        """
        Generates the JSON string that represents the network.

        Parameters
        ----------
        options : dict, optional
            `networkx` `attrs` attribute.

        keys : list, optional
            The keys to keep in the JSON string after transforming them via
            `options`.

        Returns
        -------
        str
            JSON representation of the network.
        """

        graph_data = json_graph.node_link_data(self.bn.graph, options)
        if keys:
            graph_data = {k: graph_data[k] for k in keys}

        data = {'graph': graph_data, 'parameters': self.bn.parameters,
                'data_type': self.bn.data_type}

        return json.dumps(data)

    def convert(self, io_object, options=None):
        """
        Creates the graph structure object from the JSON string representation
        received.

        Parameters
        ----------
        io_object : str

        options : dict, optional

        Returns
        -------
        BayesianNetwork
            The Bayesian Network loaded.
        """
        from ..models import BayesianNetwork

        data = json.loads(io_object)
        graph = json_graph.node_link_graph(data['graph'], directed=True,
                                           multigraph=False, attrs=options)
        parameters = data['parameters']
        data_type = data['data_type']
        return BayesianNetwork(graph=graph, parameters=parameters,
                               data_type=data_type)
