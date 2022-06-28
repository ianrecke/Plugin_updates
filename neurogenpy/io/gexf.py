"""
GEXF input/output module. It uses `networkx` functionality.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import inspect
import random

import networkx.readwrite as networkx_io
from PIL import ImageColor

from .bnio import BNIO
from .layout.dot_layout import DotLayout
from .layout.force_atlas2_layout import ForceAtlas2Layout
from .layout.igraph_layout import IgraphLayout
from .layout.image_layout import ImageLayout

_CONFIGS = {
    'small': {'maxNodeSize': 20, 'minNodeSize': 10, 'maxEdgeSize': 8,
              'minEdgeSize': 4, 'weight': 0.55, 'width': 1},
    'medium': {'maxNodeSize': 5, 'minNodeSize': 1, 'maxEdgeSize': 1.5,
               'minEdgeSize': .5, 'weight': 0.55, 'width': 1},
    'large': {'maxNodeSize': 3, 'minNodeSize': 1, 'maxEdgeSize': 0.5,
              'minEdgeSize': .1, 'weight': 1.1, 'width': 2},
    'default_color': '#FFFFFF'}


# TODO: Add to docs networkx warning: the parser uses the standard xml library
#  present in Python, which is insecure.
class GEXF(BNIO):
    """
    Bayesian network GEXF input/output class.
    """

    # TODO: Implement convert. It is not directly provided from networkx.
    def convert(self, io_object):
        """
        Creates the graph structure object from the GEXF string representation
        received.

        Returns
        -------
        networkx.DiGraph or (networkx.DiGraph, dict)
            The graph structure of the loaded Bayesian network and the
            parameters in case the format provides them.
        """
        pass

    def write_file(self, file_path='bn.gexf', layout_name=None,
                   communities=False, sizes_method='mb', layout=None):
        """
        Exports a representation of the Bayesian network structure for the
        chosen layout in GEXF format. It also calculates the sizes and colors
        of the nodes and edges of the graph according to some different
        methods.

        Parameters
        ----------
        file_path : str, default='bn.gexf'
            Path of the file to store the Bayesian network in.

        layout_name : str, optional
            Layout used for calculating the positions of the nodes in the
            graph. If it is not determined, nodes positions are not provided.

        communities : bool, default=False
            Whether to assign different colors to the nodes and edges belonging
            to different communities of Louvain.

        sizes_method : {'mb', 'neighbors'}, default='mb'
            The method used to calculate the sizes of the nodes. It can be the
            size of the Markov blanket of each node or the amount of neighbors
            they have.

        layout : dict, optional
            Custom layout to include in the GEXF representation. If
            `layout_name` is provided, it is ignored. Keys should be the nodes
            and values, tuples with the format (x_coord, y_coord).
        """

        self._add_attrs(layout_name, communities, sizes_method, layout)

        networkx_io.write_gexf(self.bn.graph, file_path)

    def read_file(self, file_path):
        return networkx_io.read_gexf(file_path)

    def generate(self, layout_name=None, communities=False,
                 sizes_method='mb', layout=None):
        """
        Generates the GEXF string that represents the network.

        Parameters
        ----------
        layout_name : str, optional
            Layout used for calculating the positions of the nodes in the
            graph. If it is not determined, nodes positions are not provided.

        communities : bool, default=False
            Whether to assign different colors to the nodes and edges belonging
            to different communities of Louvain.

        sizes_method : {'mb', 'neighbors'}, default='mb'
            The method used to calculate the sizes of the nodes. It can be the
            size of the Markov blanket of each node or the amount of neighbors
            they have.

        layout : dict, optional
            Custom layout to include in the GEXF representation. If
            `layout_name` is provided, it is ignored. Keys should be the nodes
            and values, tuples with the format (x_coord, y_coord).

        Returns
        -------
            The string representation of the network.
        """

        self._add_attrs(layout_name, communities, sizes_method, layout)

        linefeed = chr(10)
        return linefeed.join(networkx_io.generate_gexf(self.bn.graph))

    def _add_attrs(self, layout_name, communities, sizes_method, layout):
        """Add attributes related with the display of the graph."""

        layouts = {'circular': IgraphLayout, 'Dot': DotLayout,
                   'ForceAtlas2': ForceAtlas2Layout, 'Grid': IgraphLayout,
                   'fruchterman_reingold': IgraphLayout,
                   'Image': ImageLayout, 'Sugiyama': IgraphLayout}

        if layout_name is not None:
            layout = layouts[layout_name]
            params = {} if 'layout_name' not in inspect.getfullargspec(
                layout).kwonlyargs else {'layout_name': layout_name}
            layout = layout(self.bn.graph, **params).run()

        if layout is not None:
            if self.bn.num_nodes < 100:
                network_size = 'small'
            elif self.bn.num_nodes < 300:
                network_size = 'medium'
            else:
                network_size = 'large'

            nx_dict = networkx_io.json_graph.node_link_data(self.bn.graph)

            nodes_colors = self._nodes_colors(communities)
            nodes = _get_nodes_attr(nx_dict, layout,
                                    self._nodes_sizes(network_size,
                                                      method=sizes_method),
                                    nodes_colors)
            edges = _get_edges_attr(nx_dict, self._edges_sizes(network_size),
                                    self._edges_colors(nodes_colors))

            rgb_colors = {color: ImageColor.getcolor(color, 'RGB') for
                          color in set(nodes_colors.values())}
            rgb_colors["#FFFFFF"] = ImageColor.getcolor("#FFFFFF", 'RGB')

            for node in nodes:
                node_id = node['id']
                self.bn.graph.nodes[node_id]["viz"] = {"size": node['size']}
                self.bn.graph.nodes[node_id]['viz']['position'] = {
                    'x': node['x'], 'y': node['y'], 'z': 0}
                self.bn.graph.nodes[node_id]['viz']['color'] = {
                    'hex': node['color'], 'alpha': 0}
                node_color = rgb_colors[node['color']]
                self.bn.graph.nodes[node_id]['viz']['color'] = {
                    'r': node_color[0],
                    'g': node_color[1],
                    'b': node_color[2],
                    'a': 1.0}

            for edge in edges:
                x, y = edge['x'], edge['y']
                # self.bn.graph.edges[x, y]['weight'] = edge['weight']
                self.bn.graph.edges[x, y]['type'] = edge['type']
                # self.bn.graph.edges[x, y]['label'] = edge['label']
                edge_color = rgb_colors[edge['color']]
                self.bn.graph.edges[x, y]['viz'] = {
                    'color': {'r': edge_color[0], 'g': edge_color[1],
                              'b': edge_color[2], 'a': 1.0}}
                self.bn.graph.edges[x, y]['viz']['size'] = edge['size']

    def _edges_sizes(self, network_size):
        """Retrieves the size of each edge in the network."""

        edges_sizes = {}
        sum_weights = self.bn.sum_weights()
        for (x, y, edge_data) in self.bn.graph.edges(data=True):
            w_normalized = edge_data[
                               'weight'] * self.bn.num_nodes / sum_weights
            edge_size = w_normalized * _CONFIGS[network_size]['weight'] + \
                        _CONFIGS[network_size]['minEdgeSize']
            edges_sizes[(x, y)] = min(edge_size,
                                      _CONFIGS[network_size]['maxEdgeSize'])

        return edges_sizes

    def _nodes_sizes(self, network_size, method):
        """
        Retrieves the size of each node in the network according to the method
        provided.

        Parameters
        ----------
        method : {'mb', 'neighbors'}
            The method used to calculate the sizes of the nodes.

        Returns
        -------
        dict
            The size of each node and the size of the network.

        Raises
        ------
        ValueError
            If the method provided is not supported.
        """

        nodes_sizes = {}
        for node in list(self.bn.graph.nodes()):
            if method == 'mb':
                method_len = len(self.bn.markov_blanket(node))
            elif method == 'neighbors':
                method_len = len(self.bn.adjacencies(node))
            else:
                raise ValueError(f'{method} method is not supported.')

            node_size = _CONFIGS[network_size]['minNodeSize'] + method_len * \
                        _CONFIGS[network_size]['weight']

            nodes_sizes[node] = min(_CONFIGS[network_size]['maxNodeSize'],
                                    node_size)

        return nodes_sizes

    def _nodes_colors(self, communities):
        """Returns a dictionary with nodes as keys and their colors as
        values."""
        if not communities:
            return {node: _CONFIGS['default_color'] for node in
                    self.bn.graph.nodes()}

        else:
            coms = self.bn.communities()
            coms_colors = {com: _generate_random_color() for com in
                           set(coms.values())}
            return {node: coms_colors[coms[node]] for node in
                    self.bn.graph.nodes()}

    def _edges_colors(self, nodes_colors):
        """Returns a dictionary with edges as keys and their colors as
        values."""

        if len(set(nodes_colors.values())) == 1:
            return {(x, y): _CONFIGS['default_color'] for (x, y) in
                    self.bn.graph.edges()}

        edges_colors = {}
        for (x, y) in self.bn.graph.edges():
            if nodes_colors[x] == nodes_colors[y]:
                edges_colors[(x, y)] = nodes_colors[x]
            else:
                edges_colors[(x, y)] = _CONFIGS['default_color']
        return edges_colors


def _get_nodes_attr(nx_dict, layout, nodes_sizes, nodes_colors):
    """Returns the attributes of a graph according to a particular layout."""

    nodes = []

    for i, node in enumerate(nx_dict['nodes']):
        node_attributes = {
            'id': node['id'],
            'size': nodes_sizes[node['id']],
            'color': nodes_colors[node['id']],
            'x': layout[node['id']][0],
            'y': layout[node['id']][1],
        }

        nodes.append(node_attributes)

    return nodes


def _get_edges_attr(nx_dict, edges_sizes, edges_colors):
    """Returns the edges attributes."""

    edges = []
    for i, link in enumerate(nx_dict['links']):
        x = link['source']
        y = link['target']
        edge_attributes = {
            'x': x,
            'y': y,
            'type': 'directed',
            'size': edges_sizes[(x, y)],
            'color': edges_colors[(x, y)],
        }
        edges.append(edge_attributes)

    return edges


def _generate_random_color():
    """
    Generates a random RGB color with hexadecimal notation.

    Returns
    -------
    str
        A random color in hexadecimal notation.
    """

    hex_rgb = ('#%02X%02X%02X' % (
        random.randint(0, 255), random.randint(0, 255),
        random.randint(0, 255)))
    return hex_rgb
