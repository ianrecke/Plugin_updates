"""
GEXF input/output module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from datetime import datetime
from xml.dom import minidom
from xml.etree import ElementTree

import networkx.readwrite as networkx_io
from PIL import ImageColor

from .bnio import BNIO
from .layout import *

EDGE_MIN_SIZE = 0.1
EDGE_MAX_SIZE = 8
NODES_MIN_SIZE = {'small': 3, 'large': 1}
NODES_MAX_SIZE = {'small': 16, 'large': 6}
NODES_WEIGHT = {'small': 0.55, 'large': 1.1}


# TODO: Adjust sigmajs configuration
class GEXF(BNIO):
    """
    Bayesian network GEXF input/output class.
    """

    def write_file(self, bn, file_path='bn.gexf', layout_name=None):
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
        """

        if layout_name is None:
            networkx_io.write_gexf(bn.graph, file_path)

        else:
            layout_class = globals()[f'{layout_name}Layout']
            layout = layout_class(bn.graph).run()

            nx_dict = networkx_io.json_graph.node_link_data(bn.graph)

            nodes = get_nodes_attr(nx_dict, layout, _nodes_sizes(bn))
            edges = get_edges_attr(nx_dict, _edges_sizes(bn))

            gexf = ElementTree.Element('gexf')
            gexf.set('xmlns', 'http://gexf.net/1.3')
            gexf.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            gexf.set('xsi:schemaLocation',
                     'http://gexf.net/1.3\nhttp://gexf.net/1.3/gexf.xsd')
            gexf.set('version', '1.3')
            meta = ElementTree.SubElement(gexf, 'meta')
            meta.set('lastmodifieddate', datetime.today().strftime('%Y-%m-%d'))
            creator = ElementTree.SubElement(meta, 'creator')
            creator.text = 'CIG UPM'
            description = ElementTree.SubElement(meta, 'description')
            description.text = 'Bayesian Network obtained with neurogenpy.'
            keywords = ElementTree.SubElement(meta, 'keywords')
            keywords.text = 'Bayesian network, neurogenpy'

            graph = ElementTree.SubElement(gexf, 'graph')
            nodes_entry = ElementTree.SubElement(graph, 'nodes')

            for node in nodes:
                node_entry = ElementTree.SubElement(nodes_entry, 'node')
                node_entry.set('id', node['id'])
                node_entry.set('label', node['label'])
                color = ElementTree.SubElement(node_entry, 'viz:color')
                node_color = ImageColor.getcolor(node['color'], 'RGB')
                color.set('r', str(node_color[0]))
                color.set('g', str(node_color[1]))
                color.set('b', str(node_color[2]))
                color.set('a', '0')
                position = ElementTree.SubElement(node_entry, 'viz:position')
                position.set('x', node['x'])
                position.set('y', node['y'])
                position.set('z', '0')
                size = ElementTree.SubElement(node_entry, 'viz:size')
                size.set('value', node['size'])

            edges_entry = ElementTree.SubElement(graph, 'edges')
            for edge in edges:
                edge_entry = ElementTree.SubElement(edges_entry, 'edge')
                edge_entry.set('source', edge['source'])
                edge_entry.set('target', edge['target'])
                edge_entry.set('type', edge['type'])

            et_string = ElementTree.tostring(gexf, 'utf-8')

            if file_path is None:
                file_path = f'bn_{layout_name}.gexf'
            file = open(file_path, 'w')

            dom_string = minidom.parseString(et_string)
            data = dom_string.toprettyxml(indent='  ')
            file.write(data)

    def read_file(self, file_path):
        return networkx_io.read_gexf(file_path)


# TODO: Check BayesianNetwork attributes management in _edges_sizes() and
#  _nodes_sizes()
def _edges_sizes(bn):
    """Retrieves the size of each edge in the network."""

    weight_edge_size = 1.1 if bn.num_nodes < 300 else 0.55

    edges_sizes = {}
    sum_weights = bn.sum_weights()
    for (x, y, edge_data) in bn.graph.subgraph_edges(data=True):
        w_normalized = edge_data['weight'] * bn.num_nodes / sum_weights
        edge_size = w_normalized * weight_edge_size + EDGE_MIN_SIZE
        edges_sizes[(x, y)] = min(edge_size, EDGE_MAX_SIZE)

    return edges_sizes


def _nodes_sizes(bn, method='mb'):
    """
    Retrieves the size of each node in the network according to the method
    provided.

    Parameters
    ----------
    method : {'mb', 'neighbors'}, default='mb'
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

    network_size = 'small' if bn.num_nodes < 300 else 'large'
    nodes_sizes = {}
    for node in list(bn.graph.nodes()):
        if method == 'mb':
            method_len = len(bn.markov_blanket(node))
        elif method == 'neighbors':
            method_len = len(bn.adjacencies(node))
        else:
            raise ValueError(f'{method} method is not supported.')

        node_size = NODES_MIN_SIZE[network_size] + NODES_WEIGHT[
            network_size] * method_len

        nodes_sizes[node] = min(NODES_MAX_SIZE[network_size], node_size)

    return nodes_sizes


def get_nodes_attr(nx_dict, layout, nodes_sizes):
    """Returns the attributes of a graph according to a particular layout."""

    nodes = []

    # TODO: Set different sizes depending on the network size.
    for i, node in enumerate(nx_dict['nodes']):
        node_attributes = {
            'id': node['id'],
            'label': node['id'],
            'size': nodes_sizes[node],
            'color': '#000066',
            'x': layout[node['id']][0],
            'y': layout[node['id']][1],
        }

        nodes.append(node_attributes)

    return nodes


def get_edges_attr(nx_dict, edges_sizes):
    """Returns the edges attributes."""

    edges = []
    for i, link in enumerate(nx_dict['links']):
        x = link['source']
        y = link['target']
        edge_attributes = {
            'id': f'e{i}',
            'source': x,
            'target': y,
            'type': 'directed',
            'size': edges_sizes[(x, y)],
            'color': '#282c34',
            'label': str(link['weight']),
        }
        edges.append(edge_attributes)

    return edges

    # def get_sigmajs(self):
    #     self.structures_data = {}
    #     if hasattr(bn, 'structures_data'):
    #         self.structures_data = bn.structures_data
    #     self.weights_info = None
    #     if hasattr(bn, 'weights_info'):
    #         self.weights_info = bn.weights_info
    #
    #     self.sigmajs_default_settings = {
    #         # Camera options
    #         "autoRescale": True,
    #         "zoomingRatio": 1.6,
    #         "zoomMin": 0,
    #         "zoomMax": 10,
    #         # Label options:
    #         "labelAlignment": "center",  # Doesn't work
    #         "labelThreshold": 0,
    #         # Nodes, edges options:
    #         "minEdgeSize": 0.01,
    #         "edgeSizeStep": 0.01,
    #         "minWeight": "",
    #         "maxWeight": "",
    #
    #         # Edge labels (only canvas)
    #         "edgeLabelSize": "proportional",
    #         "defaultEdgeLabelSize": 14,
    #
    #         "minNodeSize": 0.01,
    #         "nodeSizeStep": 0.05,
    #         "nodeOriginalColor": '#000066',  # "#008cc2",
    #         "highlightNodeColor": "#e4002b",
    #         "color_hidden": '#eeeeee',
    #         "color_common_edges": '#66ff33',
    #         "color_structure_1": '#000000',  # "#282c34",
    #         "color_structure_2": '#0066ff',  # "#669999",
    #         "minArrowSize": 12,
    #         "maxArrowSize": 12,
    #         # Graph layout scale:
    #         "width_original": 1,
    #         "width_old": 1,
    #         "width_step": 2,
    #         "width_number_options": 5,
    #         "width_max": 0,  # Selected later in this function
    #         "height_original": 1,
    #         "height_old": 1,
    #         "height_step": 2,
    #         "height_number_options": 5,
    #         "height_max": 0,  # Selected later in this function
    #         # Speed up options:
    #     }
    #     self.sigmajs_default_settings["width_max"] = \
    #         self.sigmajs_default_settings["width_number_options"] * \
    #         self.sigmajs_default_settings["width_step"]
    #     self.sigmajs_width_options = np.arange(
    #         self.sigmajs_default_settings["width_original"],
    #         self.sigmajs_default_settings["width_max"],
    #         self.sigmajs_default_settings["width_step"])
    #     self.sigmajs_default_settings["height_max"] = \
    #         self.sigmajs_default_settings["height_number_options"] * \
    #         self.sigmajs_default_settings["height_step"]
    #     self.sigmajs_height_options = np.arange(
    #         self.sigmajs_default_settings["height_original"],
    #         self.sigmajs_default_settings["height_max"],
    #         self.sigmajs_default_settings["height_step"])
    #     self.edge_config = {
    #         "type": "arrow",
    #         "size": 1,
    #         "color": "#282c34",
    #     }
    #
    #     self.sigmajs_default_settings["structures_info"] = {}
    #     for structure_id, structure_data, in self.structures_data.items():
    #         self.sigmajs_default_settings["structures_info"][structure_id] = {
    #             "color": structure_data["color"],
    #             "percentage_instances": round(
    #                 structure_data["percentage_instances"], 2)
    #         }
    #
    #     configs = {
    #         'small': {'maxNodeSize': 12, 'maxEdgeSize': 4, 'drawLabels': True,
    #                   'hideEdgesOnMove': False,
    #                   'width': self.sigmajs_width_options[0]},
    #         'medium': {'maxNodeSize': 5, 'maxEdgeSize': 1.5,
    #                    'drawLabels': False, 'hideEdgesOnMove': False,
    #                    'width': self.sigmajs_width_options[0]},
    #         'large': {'maxNodeSize': 3, 'maxEdgeSize': 0.5,
    #                   'drawLabels': False, 'hideEdgesOnMove': True,
    #                   'width': self.sigmajs_width_options[3]}}
    #
    #     self.graph_size = list(configs.keys())[
    #         min(2, -1 * (- ((len(self.graph.nodes()) - 1) // 100) // 2))]
    #
    #     self.graph_initial_width = configs[self.graph_size]['width']
    #     self.sigmajs_default_settings["drawLabels"] = configs[self.graph_size][
    #         'drawLabels']
    #     self.sigmajs_default_settings["hideEdgesOnMove"] = \
    #         configs[self.graph_size]['drawLabels']
    #     self.sigmajs_default_settings["maxNodeSize"] = \
    #         configs[self.graph_size]['maxNodeSize']
    #     self.sigmajs_default_settings["maxEdgeSize"] = \
    #         configs[self.graph_size]['maxEdgeSize']
    #
    #     self.node_feature_config = {
    #         "size": self.sigmajs_default_settings["maxNodeSize"],
    #         "color": self.sigmajs_default_settings["nodeOriginalColor"],
    #     }
    #
    #     self.node_class_config = {
    #         "size": self.sigmajs_default_settings["maxNodeSize"],
    #         "color": "#00cc66",
    #     }
    #
    #     result = {}
    #
    #     graph_sigmajs = {
    #         "nodes": self.nodes,
    #         "edges": self.edges,
    #     }
    #
    #     if self.additional_parameters is not None:
    #         result["additional_discrete_features"] = list(
    #             self.additional_parameters["discrete_features"].keys())
    #     result["nodes"] = list(self.graph.nodes())
    #     result["nodes"].sort()
    #
    #     if self.weights_info is not None:
    #         self.sigmajs_default_settings["minWeight"] = round(
    #             self.weights_info["min"], 3)
    #         self.sigmajs_default_settings["maxWeight"] = round(
    #             self.weights_info["max"], 3)
    #
    #     self.sigmajs_default_settings["num_nodes"] = len(self.graph.nodes())
    #     self.sigmajs_default_settings[
    #         "num_edges"] = self.graph.number_of_edges()
    #
    #     result["sigmajs_default_settings"] = self.sigmajs_default_settings
    #     result["graph_sigmajs"] = graph_sigmajs
    #
    #     return result
    #
    # def get_nodes_pos(self):
    #     graph_networkx_dict = networkx_io.json_graph.node_link_data(self.graph)
    #
    #     edges = {}
    #     for i, link in enumerate(graph_networkx_dict["links"]):
    #         try:
    #             structure_id = \
    #                 self.graph.get_edge_data(link['source'], link['target'])[
    #                     'structure_id']
    #             if isinstance(structure_id, list):
    #                 edge_color = self.edge_config["color"]
    #             else:
    #                 if structure_id == 0:
    #                     color_structure_number = "color_common_edges"
    #                 else:
    #                     color_structure_number = \
    #                         f"color_structure_{structure_id}"
    #                 edge_color = self.sigmajs_default_settings[
    #                     color_structure_number]
    #
    #         except TypeError as _:
    #             structure_id = -1
    #             edge_color = self.edge_config["color"]
    #
    #         edge_id = f"e{i}"
    #         edge_sigmajs = {
    #             "id": edge_id,
    #             "source": link['source'],
    #             "target": link['target'],
    #             "type": self.edge_config["type"],
    #             "size": self.edge_config["size"],
    #             "color": edge_color,
    #             "label": str(link["weight"]),
    #             "structure_id": structure_id,
    #         }
    #         edges[edge_id] = edge_sigmajs
    #
    #     return self.layout, edges
