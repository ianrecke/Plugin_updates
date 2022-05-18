"""
GEXF input/output module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:
import inspect
import random

import networkx.readwrite as networkx_io
from PIL import ImageColor

from .bnio import BNIO
from .layout.dot_layout import DotLayout
from .layout.force_atlas2_layout import ForceAtlas2Layout
from .layout.igraph_layout import IgraphLayout
from .layout.image_layout import ImageLayout

CONFIGS = {
    'small': {'maxNodeSize': 12, 'minNodeSize': 2, 'maxEdgeSize': 8,
              'minEdgeSize': 4, 'weight': 0.55, 'width': 1},
    'medium': {'maxNodeSize': 5, 'minNodeSize': 1, 'maxEdgeSize': 1.5,
               'minEdgeSize': .5, 'weight': 0.55, 'width': 1},
    'large': {'maxNodeSize': 3, 'minNodeSize': 1, 'maxEdgeSize': 0.5,
              'minEdgeSize': .1, 'weight': 1.1, 'width': 2},
    'default_color': '#282c34'}


class GEXF(BNIO):
    """
    Bayesian network GEXF input/output class.
    """

    def write_file(self, bn, file_path='bn.gexf', layout_name=None,
                   communities=False, sizes_method='mb'):
        """
        Exports a representation of the Bayesian network for the chosen layout
        in GEXF format.

        Parameters
        ----------
        bn : BayesianNetwork
            Bayesian network to be stored.

        file_path : str, default='bn.gexf'
            Path of the file to store the Bayesian network in.

        layout_name : str, optional
            Layout used for calculating the positions of the nodes in the
            graph.

        communities : bool, default=False
            Whether to assign different colors to the nodes and edges belonging
            to different communities of Louvain.

        sizes_method : {'mb', 'neighbors'}, default='mb'
            The method used to calculate the sizes of the nodes. It can be the
            size of the Markov blanket of each node or the amount of neighbors
            they have.
        """

        layouts = {'circular': IgraphLayout, 'Dot': DotLayout,
                   'ForceAtlas2': ForceAtlas2Layout, 'Grid': IgraphLayout,
                   'FruchtermanReingold': IgraphLayout,
                   'Image': ImageLayout, 'Sugiyama': IgraphLayout}

        if layout_name is not None:
            if bn.num_nodes < 100:
                network_size = 'small'
            elif bn.num_nodes < 300:
                network_size = 'medium'
            else:
                network_size = 'large'

            layout = layouts[layout_name]
            params = {} if 'layout_name' not in inspect.getfullargspec(
                layout).kwonlyargs else {'layout_name': layout_name}
            layout = layout(bn.graph, **params).run()

            nx_dict = networkx_io.json_graph.node_link_data(bn.graph)

            nodes_colors = _nodes_colors(bn, communities)
            nodes = _get_nodes_attr(nx_dict, layout,
                                    _nodes_sizes(bn, network_size,
                                                 method=sizes_method),
                                    nodes_colors)
            edges = _get_edges_attr(nx_dict, _edges_sizes(bn, network_size),
                                    _edges_colors(bn, nodes_colors))

            rgb_colors = {color: ImageColor.getcolor(color, 'RGB') for
                          color in set(nodes_colors.values())}

            for node in nodes:
                node_id = node['id']
                bn.graph.nodes[node_id]["viz"] = {"size": node['size']}
                bn.graph.nodes[node_id]['viz']['position'] = {
                    'x': node['x'], 'y': node['y'], 'z': 0}
                bn.graph.nodes[node_id]['viz']['color'] = {
                    'hex': node['color'], 'alpha': 0}
                node_color = rgb_colors[node['color']]
                bn.graph.nodes[node_id]['viz']['color'] = {'r': node_color[0],
                                                           'g': node_color[1],
                                                           'b': node_color[2],
                                                           'a': 0}

            for edge in edges:
                x, y = edge['x'], edge['y']
                bn.graph.edges[x, y]['weight'] = edge['weight']
                bn.graph.edges[x, y]['type'] = edge['type']
                # bn.graph.edges[x, y]['label'] = edge['label']
                edge_color = rgb_colors[edge['color']]
                bn.graph.edges[x, y]['viz'] = {
                    'color': {'r': edge_color[0], 'g': edge_color[1],
                              'b': edge_color[2], 'a': 0}}

        networkx_io.write_gexf(bn.graph, file_path)

    def read_file(self, file_path):
        return networkx_io.read_gexf(file_path)


def _edges_sizes(bn, network_size):
    """Retrieves the size of each edge in the network."""

    edges_sizes = {}
    sum_weights = bn.sum_weights()
    for (x, y, edge_data) in bn.graph.edges(data=True):
        w_normalized = edge_data['weight'] * bn.num_nodes / sum_weights
        edge_size = w_normalized * CONFIGS[network_size]['weight'] + \
                    CONFIGS[network_size]['minEdgeSize']
        edges_sizes[(x, y)] = min(edge_size,
                                  CONFIGS[network_size]['maxEdgeSize'])

    return edges_sizes


def _nodes_sizes(bn, network_size, method):
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
    for node in list(bn.graph.nodes()):
        if method == 'mb':
            method_len = len(bn.markov_blanket(node))
        elif method == 'neighbors':
            method_len = len(bn.adjacencies(node))
        else:
            raise ValueError(f'{method} method is not supported.')

        node_size = CONFIGS[network_size]['minNodeSize'] + method_len * \
                    CONFIGS[network_size]['weight']

        nodes_sizes[node] = min(CONFIGS[network_size]['maxNodeSize'],
                                node_size)

    return nodes_sizes


def _nodes_colors(bn, communities):
    """Returns a dictionary with nodes as keys and their colors as values."""
    if not communities:
        return {node: CONFIGS['default_color'] for node in bn.graph.nodes()}

    else:
        coms = bn.communities()
        coms_colors = {com: _generate_random_color() for com in
                       set(coms.values())}
        return {node: coms_colors[coms[node]] for node in bn.graph.nodes()}


def _edges_colors(bn, nodes_colors):
    """Returns a dictionary with edges as keys and their colors as values."""

    if len(set(nodes_colors.values())) == 1:
        return {(x, y): CONFIGS['default_color'] for (x, y) in
                bn.graph.edges()}

    edges_colors = {}
    for (x, y) in bn.graph.edges():
        if nodes_colors[x] == nodes_colors[y]:
            edges_colors[(x, y)] = nodes_colors[x]
        else:
            edges_colors[(x, y)] = CONFIGS['default_color']
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
            'weight': edges_sizes[(x, y)],
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
