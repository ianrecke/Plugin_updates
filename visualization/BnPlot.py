from .BnLayout import BnLayout

import networkx
import networkx.readwrite as networkx_io
import numpy as np
import plotly as plotly_graph
from deprecated import deprecated


class BnPlot(object):
    """
    Bayesian network visualization class.
    """
    def __init__(self, bn):
        """
        Bayesian network visualization class constructor.
        @param bn: a Bayesian network to be visualized.
        """
        self.graph = bn.graph
        self.features_classes = bn.features_classes
        self.additional_parameters = bn.additional_parameters
        self.structures_data = {}
        if hasattr(bn, 'structures_data'):
            self.structures_data = bn.structures_data
        self.weights_info = None
        if hasattr(bn, 'weights_info'):
            self.weights_info = bn.weights_info

        self.sigmajs_default_settings = {
            # Camera options
            "autoRescale": True,
            "zoomingRatio": 1.6,
            "zoomMin": 0,
            "zoomMax": 10,
            # Label options:
            "labelAlignment": "center",  # Doesn't work
            "labelThreshold": 0,
            # Nodes, edges options:
            "minEdgeSize": 0.01,
            "maxEdgeSize": 4,
            "edgeSizeStep": 0.01,
            "minWeight": "",
            "maxWeight": "",

            # --Edge labels--- (only canvas
            "edgeLabelSize": "proportional",
            "defaultEdgeLabelSize": 14,

            # ----Enable edges click------ (only canvas):
            # "enableEdgeHovering":  True,
            # "edgeHoverColor":  "red",
            # 'defaultEdgeHoverColor': 'red',
            # 'edgeHoverSizeRatio': 2,

            "minNodeSize": 0.01,
            "maxNodeSize": 12,
            "nodeSizeStep": 0.05,
            "nodeOriginalColor": '#000066',  # "#008cc2",
            "highlightNodeColor": "#e4002b",
            "color_hidden": '#eeeeee',
            "color_common_edges": '#66ff33',
            "color_structure_1": '#000000',  # "#282c34",
            "color_structure_2": '#0066ff',  # "#669999",
            "minArrowSize": 12,
            "maxArrowSize": 12,
            # Graph layout scale:
            "width_original": 1,
            "width_old": 1,
            "width_step": 2,
            "width_number_options": 5,
            "width_max": 0,  # Selected later in this function
            "height_original": 1,
            "height_old": 1,
            "height_step": 2,
            "height_number_options": 5,
            "height_max": 0,  # Selected later in this function
            # Speed up options:
            "drawLabels": True,  # Selected later in this function
            "hideEdgesOnMove": False  # Selected later in this function
        }
        self.sigmajs_default_settings["width_max"] = \
            (self.sigmajs_default_settings["width_number_options"] *
             self.sigmajs_default_settings["width_step"])
        self.sigmajs_width_options = np.arange(
            self.sigmajs_default_settings["width_original"],
            self.sigmajs_default_settings["width_max"],
            self.sigmajs_default_settings["width_step"])
        self.sigmajs_default_settings["height_max"] = \
            (self.sigmajs_default_settings["height_number_options"] *
             self.sigmajs_default_settings["height_step"])
        self.sigmajs_height_options = np.arange(
            self.sigmajs_default_settings["height_original"],
            self.sigmajs_default_settings["height_max"],
            self.sigmajs_default_settings["height_step"])
        self.edge_config = {
            "type": "arrow",
            "size": 1,
            "color": "#282c34",
        }

        self.sigmajs_default_settings["structures_info"] = {}
        for structure_id, structure_data, in self.structures_data.items():
            self.sigmajs_default_settings["structures_info"][structure_id] = {
                "color": structure_data["color"],
                "percentage_instances": round(
                    structure_data["percentage_instances"], 2)
            }

        self.graph_type_size = "small"
        self.graph_initial_height = self.sigmajs_width_options[0]
        self.graph_initial_height = self.sigmajs_height_options[0]
        if len(self.graph.nodes()) > 300:
            self.graph_type_size = "large"
            self.sigmajs_default_settings["maxNodeSize"] = 3
            self.sigmajs_default_settings["maxEdgeSize"] = 0.5
            self.graph_initial_height = self.sigmajs_width_options[3]
            self.graph_initial_height = self.sigmajs_height_options[0]
        elif len(self.graph.nodes()) > 100:
            self.graph_type_size = "medium"
            self.sigmajs_default_settings["maxNodeSize"] = 5
            self.sigmajs_default_settings["maxEdgeSize"] = 1.5
            self.graph_initial_height = self.sigmajs_width_options[0]
            self.graph_initial_height = self.sigmajs_height_options[0]

        self.node_feature_config = {
            "size": self.sigmajs_default_settings["maxNodeSize"],
            "color": self.sigmajs_default_settings["nodeOriginalColor"],
        }

        self.node_class_config = {
            "size": self.sigmajs_default_settings["maxNodeSize"],
            "color": "#00cc66",
        }

    def draw_networkx_sigmajs(self, layout_name="circular",
                              additional_params=None):
        if layout_name is None:
            layout_name = "circular"
        result = {}

        graph_sigmajs = {
            "nodes": [],
            "edges": [],
        }

        graph_networkx_dict = networkx_io.json_graph.node_link_data(self.graph)

        # TODO: Instantiate appropriate Layout class
        bn_layout = BnLayout.get_layout_class(layout_name, self.graph,
                                              self.graph_initial_height,
                                              self.graph_initial_height,
                                              additional_params)
        layout = bn_layout.run()

        for i, link in enumerate(graph_networkx_dict["links"]):
            try:
                structure_id = \
                    self.graph.get_edge_data(link['source'], link['target'])[
                        'structure_id']
                if isinstance(structure_id, list):
                    edge_color = self.edge_config["color"]
                else:
                    if structure_id == 0:
                        color_structure_number = "color_common_edges"
                    else:
                        color_structure_number = "color_structure_{}".format(
                            structure_id)
                    edge_color = self.sigmajs_default_settings[
                        color_structure_number]

            except Exception as e:
                structure_id = -1
                edge_color = self.edge_config["color"]

            edge_sigmajs = {
                "id": "e{0}".format(i),
                "source": link['source'],
                "target": link['target'],
                "type": self.edge_config["type"],
                "size": self.edge_config["size"],
                "color": edge_color,
                "label": str(link["weight"]),
                "structure_id": structure_id,
            }
            graph_sigmajs["edges"].append(edge_sigmajs)

        if self.weights_info is not None:
            self.sigmajs_default_settings["minWeight"] = round(
                self.weights_info["min"], 3)
            self.sigmajs_default_settings["maxWeight"] = round(
                self.weights_info["max"], 3)

        for i, node in enumerate(graph_networkx_dict["nodes"]):
            if node["id"] in self.features_classes:
                node_config = self.node_class_config
            else:
                node_config = self.node_feature_config

            node_sigmajs = {
                "id": node["id"],
                "label": node["id"],
                "size": node_config["size"],
                "color": node_config["color"],
                "x": layout[node["id"]][0],
                "y": layout[node["id"]][1],
            }

            """
            if self.additional_parameters is not None and node["id"] in self.additional_parameters["nodes"]:
                #If node has discrete features:
                if self.additional_parameters["nodes"][node["id"]]["discrete_features"]:
                    node_features = self.additional_parameters["nodes"][node["id"]]["discrete_features"]
                    node_feature_selected = list(node_features.keys())[0]#The user can change it in the UI
                    node_category_selected = node_features[node_feature_selected][0]

                    color_feature = self.additional_parameters["discrete_features"][node_feature_selected][node_category_selected]["color"]
                    node_sigmajs["color"] = color_feature
                    node_sigmajs["label"] += " | ({0})".format(node_category_selected)
            """
            graph_sigmajs["nodes"].append(node_sigmajs)

        if self.additional_parameters is not None:
            result["additional_discrete_features"] = list(
                self.additional_parameters["discrete_features"].keys())
        result["nodes"] = list(self.graph.nodes())
        result["nodes"].sort()

        self.sigmajs_default_settings["num_nodes"] = len(self.graph.nodes())
        self.sigmajs_default_settings[
            "num_edges"] = self.graph.number_of_edges()

        if self.graph_type_size != "small":
            self.sigmajs_default_settings["drawLabels"] = False
        if self.graph_type_size == "large":
            self.sigmajs_default_settings["hideEdgesOnMove"] = True
        result["sigmajs_default_settings"] = self.sigmajs_default_settings
        result["graph_sigmajs"] = graph_sigmajs

        return result

    def get_layouts_nodes_pos(self, layout_name="circular",
                              additional_params=None):
        if layout_name is None:
            layout_name = "circular"

        bn_layout = BnLayout.get_layout_class(layout_name, self.graph,
                                              self.graph_initial_height,
                                              self.graph_initial_height,
                                              additional_params)
        nodes_pos = bn_layout.run()

        graph_networkx_dict = networkx_io.json_graph.node_link_data(self.graph)

        edges = {}
        for i, link in enumerate(graph_networkx_dict["links"]):
            try:
                structure_id = \
                    self.graph.get_edge_data(link['source'], link['target'])[
                        'structure_id']
                if isinstance(structure_id, list):
                    edge_color = self.edge_config["color"]
                else:
                    if structure_id == 0:
                        color_structure_number = "color_common_edges"
                    else:
                        color_structure_number = "color_structure_{}".format(
                            structure_id)
                    edge_color = self.sigmajs_default_settings[
                        color_structure_number]

            except Exception as e:
                structure_id = -1
                edge_color = self.edge_config["color"]

            edge_id = "e{0}".format(i)
            edge_sigmajs = {
                "id": edge_id,
                "source": link['source'],
                "target": link['target'],
                "type": self.edge_config["type"],
                "size": self.edge_config["size"],
                "color": edge_color,
                "label": str(link["weight"]),
                "structure_id": structure_id,
            }
            edges[edge_id] = edge_sigmajs

        return nodes_pos, edges

    @deprecated
    def draw_networkx_plotly(self):
        # G = nx.Graph()
        # G.add_nodes_from(self.model.V)
        # G.add_edges_from(self.model.E)
        result = {}
        G = self.model
        N = G.nodes()
        E = G.edges()
        labels_nodes = list(N)

        try:
            pos = networkx.fruchterman_reingold_layout(G)

            Xv = [pos[k][0] for k in N]
            Yv = [pos[k][1] for k in N]
            Xed = []
            Yed = []
            for edge in E:
                Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
                Yed += [pos[edge[0]][1], pos[edge[1]][1], None]

            trace3 = plotly_graph.Scatter(x=Xed,
                                          y=Yed,
                                          mode='lines',
                                          line=dict(color='rgb(210,210,210)',
                                                    width=1),
                                          hoverinfo='none'
                                          )
            trace4 = plotly_graph.Scatter(
                x=Xv,
                y=Yv,
                mode='markers+text',
                name='net',
                marker=dict(
                    symbol='circle-dot',
                    size=18,
                    color='#6959CD',
                    line=dict(
                        color='rgb(50,50,50)',
                        width=0.5)),
                text=labels_nodes,
                textposition="top center",
                textfont=dict(
                    size=18,
                ),
                hoverinfo='text')

            annot = "Bayesian network"

            data1 = [trace3, trace4]

            axis = dict(showline=False,
                        # hide axis line, grid, ticklabels and  title
                        zeroline=False,
                        showgrid=False,
                        showticklabels=False,
                        title=''
                        )

            height = 800
            """
            annotations = []
            for i in range(0, len(x0)):
                annotations = [
                    dict(ax=x0[i], ay=y0[i], axref='x', ayref='y',
                         x=x1[i], y=y1[i], xref='x', yref='y')
                ]
            """
            annotations = [dict()]
            layout = plotly_graph.Layout(
                title="Bayesian network graph",
                font=dict(size=12),
                showlegend=False,
                autosize=True,
                height=height,
                xaxis=plotly_graph.layout.XAxis(axis),
                yaxis=plotly_graph.layout.YAxis(axis),
                margin=plotly_graph.layout.Margin(
                    l=40,
                    r=40,
                    b=85,
                    t=100,
                ),
                hovermode='closest',
                annotations=annotations
            )
            figure = plotly_graph.Figure(data=data1, layout=layout)
            figure['layout']['annotations'][0]['text'] = annot

            result["object_result_html"] = plotly_graph.offline.plot(
                figure, include_plotlyjs=False, show_link=False,
                output_type='div')

        except Exception as e:
            result["error"] = True
            result["error_message"] = str(e)

        return result
