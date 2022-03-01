import os
from django.conf import settings
import random
import copy
import numpy as np
from operator import itemgetter
import networkx
import networkx.algorithms.centrality.betweenness as betweenness
from community import best_partition

from ..learn_parameters import GaussianNode
from ..learn_structure import CL, FastIamb, FGES, Genie, Glasso, Gs, Hc, \
    HcTabu, HitonPC, Iamb, InterIamb, Lr, MBC, MiContinuous, MMHC, MMPC, NB, \
    PC, Pearson, SparseBn, Tan
from ..helpers.helpers import generate_random_color, dataframe_get_type, \
    density_functions_bn
from ..inference import Evidence, GaussianExact

_ESTIMATORS = ['DiscreteBE', 'DiscreteMLE', 'GaussianMLE']


def _check_estimator(estimation):
    if estimation not in _ESTIMATORS:
        raise AttributeError("Parameters learning is only available for the "
                             f"following estimation methods: {*_ESTIMATORS,}.")
    return True


# TODO: Functions or attributes? Decide
class BayesianNetwork:
    """
    Bayesian network class.
    """

    def __init__(self, graph=None, parameters=None, joint_dist=None,
                 evidence=None):
        """
        Bayesian network class constructor.
        @param graph: Graph structure of the network.
        @param parameters: Parameters of the graph structure.
        @param joint_dist:
        @param evidence:
        """
        self.graph = graph
        self.parameters = parameters
        self.topological_order = None
        self.evidence = Evidence()
        self.joint_dist_path = None
        self.joint_dist_cond_path = None
        self.additional_parameters = None

        self.weights_info = None
        self.topological_order = None
        if self.graph is not None and len(self.graph.edges()) > 0:
            self.init_graph_edges_weights()
            self._set_topological_order()

        self.parameters = parameters
        self.joint_dist = joint_dist  # Save it?
        self.evidence = evidence

        # To fix bnlearn string problems:
        # for i, col in enumerate(self.features_classes):
        #     self.features_classes[i] = self.feature_to_str(col)

        # if not is_uploaded_bn_file:
        # dataframe = dataset.get_dataframe()
        #
        # self.data_type = dataframe_get_type(dataframe.dtypes)
        # if self.data_type == "discrete":
        #     dataframe = dataframe.astype("str").astype("category")
        #
        # # To fix bnlearn string problems:
        # dataframe.columns = dataframe.columns.str.replace(".", "")
        # dataframe.columns = dataframe.columns.str.replace("-", "")
        # dataframe.columns = dataframe.columns.str.replace(":", "")
        # new_cols = []
        # for col in dataframe.columns:
        #     col = self.feature_to_str(col)
        #     new_cols.append(col)
        # dataframe.columns = new_cols
        #
        # self.dataset.update_and_save(dataframe)

        # self.original_graph = self.graph
        # if self.graph is not None:
        #     self.original_graph = self.graph.copy()

    def _set_topological_order(self):
        """
        Sets a topological order between nodes, i.e., if there is an edge
        u -> v between two nodes, u appears before v in the ordering.
        """
        try:
            topological_order_ids = list(networkx.topological_sort(self.graph))
            nodes_names = list(self.graph.nodes())
            topological_order = [nodes_names.index(node) for node in
                                 topological_order_ids]
        except networkx.NetworkXUnfeasible as _:
            topological_order = None

        self.topological_order = topological_order

    # def topological_sort(self):
    #     # Returns a topological ordering of the graph
    #     def sortUtil(node, visited, stack):
    #         visited[node] = True
    #         for child in self.graph.successors(node):
    #             if not visited[child]:
    #                 sortUtil(child, visited, stack)
    #         node_idx = list(self.graph.nodes()).index(node)
    #         stack.insert(0, node_idx)
    #
    #     visited = {}
    #     for node in self.graph.nodes():
    #         visited[node] = False
    #     stack = []
    #     for node in self.graph.nodes():
    #         if not visited[node]:
    #             sortUtil(node, visited, stack)
    #
    #     return stack

    def init_graph_edges_weights(self):
        all_weights = []
        for (x, y, edge_data) in self.graph.edges(data=True):
            if "weight" not in edge_data:
                edge_data["weight"] = 1
            all_weights.append(edge_data["weight"])

        all_weights = np.array(all_weights, dtype=np.float64)
        self.weights_info = {
            "sum": np.sum(all_weights),
            "min": np.amin(all_weights),
            "max": np.max(all_weights),
            "mean": all_weights.mean(),
            "var": all_weights.var()
        }

    def add_new_graph_edges_net_attr(self, new_graph, structure_id):
        for (x, y) in new_graph.edges():
            if not self.graph.has_edge(x, y):
                self.graph.add_edge(x, y)
            edge_data = self.graph.get_edge_data(x, y)
            if "weight" not in edge_data:
                edge_data["weight"] = 1
            if "structure_id" not in edge_data:
                edge_data["structure_id"] = structure_id
            else:
                edge_data["structure_id"] = 0

    def add_new_multi_graph_edges_net_attr(self, new_graph, structure_id):
        for (x, y) in new_graph.edges():
            if not self.graph.has_edge(x, y):
                self.graph.add_edge(x, y)
            edge_data = self.graph.get_edge_data(x, y)
            if "weight" not in edge_data:
                edge_data["weight"] = 1
            if "structure_id" not in edge_data:
                edge_data["structure_id"] = [structure_id]
            else:
                edge_data["structure_id"].append(structure_id)

    def get_network_stats(self):
        """
        Returns the number of nodes and edges of the Bayesian network.
        @return: A tuple with the number of nodes and edges.
        """
        num_nodes = len(self.graph.nodes())
        num_edges = self.graph.number_of_edges()

        return num_nodes, num_edges

    def get_random_continuous_params(self):
        model_parameters = {}
        for node in self.graph.nodes():
            parents_names = list(self.graph.predecessors(node))

            if len(parents_names) == 0:
                mean = np.random.randint(-10, 10)
                variance = np.random.randint(1, 10)
                parents_coeffs = []
            else:
                mean = np.random.randint(-10, 10)
                variance = np.random.randint(1, 10)
                parents_coeffs = np.random.rand(len(parents_names))

            model_parameters[node] = GaussianNode(mean,
                                                  variance,
                                                  parents_names,
                                                  parents_coeffs)

        return model_parameters

    @staticmethod
    def get_saved_bn_by_model_name(data_client_json, session):
        if "model_name" in data_client_json:
            model_name = data_client_json["model_name"]
        else:
            model_name = "ml_bayesian_network"
        bn = session[model_name]

        return bn

    def feature_to_str(self, feature):
        str_result = feature
        if feature.isdigit():
            str_result = f"x_{feature}_"

        return str_result

    def relabel_nodes_names(self, new_nodes_names):
        mapping = {}
        for i, node in enumerate(self.graph.nodes()):
            mapping[node] = new_nodes_names[i]

        graph = networkx.relabel_nodes(self.graph, mapping)

        return graph

    def set_additional_parameters(self, additional_parameters):
        mapping_alt_names = {}
        for group_key, group_val in additional_parameters[
            "discrete_features"].items():
            for category_key, category_val in group_val.items():
                if "color" not in category_val:
                    random_color_hex = generate_random_color()
                    category_val["color"] = random_color_hex

        nodes_names = list(self.graph.nodes())
        additional_parameters_nodes = list(
            additional_parameters["nodes"].items())

        for node_key, node_val in additional_parameters_nodes:
            if "alternative_name" not in node_val:
                mapping_alt_names[node_key] = node_key
                continue
            alt_name = node_val["alternative_name"]

            if node_key in nodes_names:
                mapping_alt_names[node_key] = node_val["alternative_name"]
                node_val["alternative_name"] = node_key
            elif node_val["alternative_name"] in nodes_names:
                mapping_alt_names[alt_name] = alt_name

            del additional_parameters["nodes"][node_key]
            additional_parameters["nodes"][alt_name] = node_val
            if self.additional_parameters:
                self.additional_parameters["nodes"][alt_name][
                    "discrete_features"].update(node_val["discrete_features"])

        if mapping_alt_names:
            self.graph = networkx.relabel_nodes(self.graph, mapping_alt_names)

        if self.additional_parameters:
            self.additional_parameters["discrete_features"].update(
                additional_parameters["discrete_features"])

        if not self.additional_parameters:
            self.additional_parameters = additional_parameters

        return 0

    def set_random_additional_parameters(self):
        discrete_features = {
            "brain_region": {
                "hippocampus": {
                    "color": "#14cb72"
                },
                "visual_cortex": {
                    "color": "#98ca12"
                },
                "hypothalamus": {
                    "color": "#511a85"
                }
            },
            "disease": {
                "parkinson": {
                    "color": "#401b72"
                },
                "alzheimer": {
                    "color": "#81ca12"
                }
            }
        }
        additional_parameters = {
            "miscellaneous_parameters": {
                "prob_params_type": "continuous"
            },
            "discrete_features": discrete_features,
            "nodes": {
            },
        }

        groups = list(discrete_features.keys())

        for node in self.graph.nodes():
            node_config = {
                "prob_params": {
                    "mean": 0.5,
                    "sd": 1.2
                },
                "discrete_features": {
                }
            }
            for group_i in groups:
                categories_group = list(discrete_features[group_i].keys())
                random_num_categories = random.randint(1,
                                                       len(categories_group))
                random_categories = random.sample(set(categories_group),
                                                  random_num_categories)

                node_config["discrete_features"][group_i] = random_categories

            additional_parameters["nodes"][node] = node_config

        self.set_additional_parameters(additional_parameters)

        return 0

    def get_markov_blanket(self, node_id):
        """
        Returns the Markov blanket of a node.
        @param node_id: ID of the node which Markov blanket have to be
            retrieved.
        @return: A list with the IDs of the nodes that form the chosen node's
            Markov blanket.
        """
        markov_blanket = [node_id]

        graph = self.graph

        parents = list(graph.predecessors(node_id))
        children = list(graph.successors(node_id))
        markov_blanket += parents + children
        parents_children = []
        for child in children:
            parents_child = graph.predecessors(child)
            for parent_child in parents_child:
                if parent_child not in markov_blanket:
                    parents_children.append(parent_child)

        markov_blanket += parents_children

        return markov_blanket

    def get_parents(self, node_id):
        """
        Returns the parents of a node.
        @param node_id: ID of the node which parents have to be retrieved.
        @return: A list with the IDs of the chosen node's parents.
        """
        parents = list(self.graph.predecessors(node_id))

        return parents

    def get_children(self, node_id):
        """
        Returns the children of a node.
        @param node_id: ID of the node which children have to be retrieved.
        @return: A list with the IDs of the chosen node's children.
        """
        children = list(self.graph.successors(node_id))

        return children

    def get_direct_neighbors(self, node_id):
        """
        Returns the direct neighbors (parents and children) of a node.
        @param node_id: ID of the node which direct neighbors have to be
            retrieved.
        @return: A list with the IDs of the chosen node's direct neighbors.
        """
        parents = list(self.graph.predecessors(node_id))
        children = list(self.graph.successors(node_id))

        direct_neighbors = parents + children

        return direct_neighbors

    def get_nodes_sizes_by_markov_blanket(self):
        if len(self.graph.nodes()) < 300:
            min_node_size = 3
            max_node_size = 16
            weight_node_size = 1.1
        else:
            min_node_size = 1
            max_node_size = 6
            weight_node_size = 0.55
        nodes_sizes = {}
        for node in self.graph.nodes():
            nodes_sizes[node] = self.node_size_by_num_markov_blanket(node,
                                                                     min_node_size,
                                                                     max_node_size,
                                                                     weight_node_size)

        return nodes_sizes, min_node_size, max_node_size

    def node_size_by_num_markov_blanket(self, node_id, min_node_size,
                                        max_node_size, weight_node_size):
        num_nodes_mv = len(self.get_markov_blanket(node_id))
        node_size = num_nodes_mv * weight_node_size + min_node_size

        if node_size >= max_node_size:
            node_size = max_node_size

        return node_size

    def get_additional_parameters(self):
        return self.additional_parameters

    def add_new_additional_parameters(self, communities_nodes, com,
                                      custom_group_name="group ",
                                      group_color=None):
        communities_groups = {}
        communities_colors = {}
        groups_number = 1

        for node in communities_nodes:
            if communities_nodes[node] not in communities_colors.values():
                if custom_group_name == "group ":
                    communities_colors[
                        custom_group_name + str(groups_number)] = \
                        communities_nodes[node]
                    groups_number += 1
                    communities_groups[node] = custom_group_name + str(
                        len(communities_colors))
                else:
                    communities_colors[custom_group_name] = {
                        'color': group_color}
                    communities_groups[node] = custom_group_name
            else:
                communities_colors_values = list(communities_colors.values())
                group_index = 0
                found = False
                while not found:
                    if communities_colors_values[group_index] == \
                            communities_nodes[node]:
                        found = True
                    group_index += 1
                if custom_group_name == "group ":
                    group_name = custom_group_name + str(group_index)
                else:
                    group_name = custom_group_name
                communities_groups[node] = group_name

        additional_parameters = self.get_additional_parameters()
        if additional_parameters is None:
            additional_parameters = {'discrete_features': {}, 'nodes': {}}
            for node in communities_groups:
                additional_parameters['nodes'][node] = {
                    'discrete_features': {}}
        if com in additional_parameters['discrete_features']:
            additional_parameters['discrete_features'][com].update(
                communities_colors)
        else:
            additional_parameters['discrete_features'][
                com] = communities_colors

        for node in communities_groups:
            if node in additional_parameters['nodes']:
                if com in additional_parameters['nodes'][node][
                    'discrete_features']:
                    variable = \
                        additional_parameters['nodes'][node][
                            'discrete_features'][
                            com]
                    variable.append(communities_groups[node])
                else:
                    additional_parameters['nodes'][node]['discrete_features'][
                        com] = [communities_groups[node]]
            else:
                additional_parameters['nodes'][node] = {
                    'discrete_features': {}}
                additional_parameters['nodes'][node]['discrete_features'][
                    com] = [communities_groups[node]]

        return additional_parameters

    def restore_additional_parameters(self, selection_option):
        additional_parameters = self.get_additional_parameters()
        print(additional_parameters['discrete_features'][selection_option])
        additional_parameters['discrete_features'].pop(selection_option)
        for node in additional_parameters['nodes']:
            additional_parameters['nodes'][node]['discrete_features'].pop(
                selection_option)

        return additional_parameters

    def get_nodes_sizes_by_neighbors(self):
        if len(self.graph.nodes()) < 300:
            min_node_size = 3
            max_node_size = 16
            weight_node_size = 1.1
        else:
            min_node_size = 1
            max_node_size = 6
            weight_node_size = 0.55

        nodes_sizes = {}
        for node in self.graph.nodes():
            nodes_sizes[node] = self.node_size_by_num_neighbors(node,
                                                                min_node_size,
                                                                max_node_size,
                                                                weight_node_size)

        return nodes_sizes, min_node_size, max_node_size

    def node_size_by_num_neighbors(self, node_id, min_node_size, max_node_size,
                                   weight_node_size):
        num_neighbors = len(self.get_direct_neighbors(node_id))
        node_size = num_neighbors * weight_node_size + min_node_size

        if node_size >= max_node_size:
            node_size = max_node_size

        return node_size

    def get_important_nodes(self, selection_option, min_num_neighbors=0,
                            include_neighbors=0, max_node_size=0):
        important_nodes = {}
        weight_node_size = 2

        if selection_option == "degrees":
            important_nodes = self.get_important_nodes_degrees(important_nodes,
                                                               min_num_neighbors,
                                                               include_neighbors,
                                                               max_node_size,
                                                               weight_node_size)
        elif selection_option == "betweenness-centrality":
            important_nodes = self.get_important_nodes_betweenness(
                important_nodes, max_node_size)

        return important_nodes

    def get_important_nodes_degrees(self, important_nodes, min_num_neighbors,
                                    include_neighbors, max_node_size,
                                    weight_node_size):
        for node_id in self.graph.nodes():
            neighbors = self.get_direct_neighbors(node_id)
            num_neighbors = len(neighbors)
            if num_neighbors >= min_num_neighbors:
                important_nodes[node_id] = {
                    "color": generate_random_color(),
                    "size": max_node_size * weight_node_size,
                }
                if include_neighbors:
                    important_nodes[node_id]["neighbors"] = neighbors

        return important_nodes

    def get_important_nodes_betweenness(self, important_nodes, max_node_size):
        nodes_importance = betweenness.betweenness_centrality(self.graph)
        nodes_number = len(self.graph.nodes)
        weight_node_size = max_node_size * nodes_number
        new_max_node_size = max_node_size + 15

        for node in nodes_importance:
            if nodes_importance[node] != 0:
                node_size = nodes_importance[
                                node] * weight_node_size + max_node_size
            else:
                node_size = 0
            if node_size >= new_max_node_size:
                node_size = new_max_node_size
            important_nodes[node] = {
                "color": generate_random_color(),
                "size": node_size
            }

        return important_nodes

    def get_communities(self, selection_option):
        communities = {}
        elements_number_by_group = {}
        important_communities = {}

        if selection_option == "louvain":
            communities = self.get_communities_louvain(communities)

        for community in communities:
            if communities[community][
                'color'] not in elements_number_by_group.keys():
                elements_number_by_group[communities[community]['color']] = 1
            else:
                elements_number_by_group[communities[community]['color']] = \
                    elements_number_by_group[
                        communities[community]['color']] + 1

        for community in communities:
            if elements_number_by_group[communities[community]['color']] > 1:
                important_communities[community] = communities[community]

        return important_communities

    def get_communities_louvain(self, communities):
        undirected_graph = self.graph.to_undirected()
        communities_nodes = best_partition(undirected_graph)
        communities_colors = []
        for node in communities_nodes:
            if communities_nodes[node] > len(communities_colors):
                communities[node] = {
                    "color": communities_colors[communities_nodes[node]]
                }
            else:
                communities_colors.append(generate_random_color())
                communities[node] = {
                    "color": communities_colors[communities_nodes[node]]
                }

        return communities

    def edge_size_by_weight(self, edge_weight, min_edge_size, max_edge_size,
                            weight_edge_size):
        edge_weight_normalized = edge_weight * len(self.graph.nodes()) / \
                                 self.weights_info["sum"]

        edge_size = edge_weight_normalized * weight_edge_size + min_edge_size

        if edge_size >= max_edge_size:
            edge_size = max_edge_size

        return edge_size

    def get_edges_sizes_by_weights(self, edges_sep_char="$$"):
        if len(self.graph.edges()) < 300:
            min_edge_size = 0.1
            max_edge_size = 8
            weight_edge_size = 1.1
        else:
            min_edge_size = 0.1
            max_edge_size = 8
            weight_edge_size = 0.55

        edges_sizes = {}
        for (x, y, edge_data) in self.graph.edges(data=True):
            edges_sizes[x + edges_sep_char + y] = self.edge_size_by_weight(
                edge_data["weight"], min_edge_size,
                max_edge_size, weight_edge_size)

        return edges_sizes, min_edge_size, max_edge_size

    def get_groups(self):
        groups = []

        if self.additional_parameters:
            groups = list(
                self.additional_parameters["discrete_features"].keys())

        return groups

    def get_info_nodes_by_group(self, group_id):
        info_nodes = {}

        if self.additional_parameters:
            if group_id in self.additional_parameters["discrete_features"]:
                categories_in_group = \
                    self.additional_parameters["discrete_features"][group_id]
                for node_key, node_val in self.additional_parameters[
                    "nodes"].items():
                    if group_id in node_val["discrete_features"]:
                        category_node = \
                            node_val["discrete_features"][group_id][
                                0]  # Multiple elements not supported yet
                        category_color = categories_in_group[category_node][
                            "color"]
                        info_nodes[node_key] = {
                            "category": category_node,
                            "color": category_color,
                        }

        return info_nodes

    def get_categories_in_group(self, group_id):
        categories_in_group = []

        if self.additional_parameters:
            if group_id in self.additional_parameters["discrete_features"]:
                categories_in_group = list(
                    self.additional_parameters["discrete_features"][
                        group_id].keys())

        return categories_in_group

    def get_category_node(self, group_id, node_id):
        category_id = None
        if self.additional_parameters:
            category_id = \
                self.additional_parameters["nodes"][node_id][
                    "discrete_features"][
                    group_id][0]

        return category_id

    def get_nodes_in_category(self, group_id, category_id, structure_id,
                              show_neighbors):
        nodes_in_category = []
        color_category = ""
        neighbors = set()

        if self.additional_parameters:
            color_category = \
                self.additional_parameters["discrete_features"][group_id][
                    category_id]["color"]

            for node_key, node_val in self.additional_parameters[
                "nodes"].items():
                if group_id in node_val["discrete_features"].keys():
                    node_group = node_val["discrete_features"][group_id]
                    if category_id == node_group[0]:
                        nodes_in_category.append(node_key)

        if show_neighbors:
            for node_id in nodes_in_category:
                neighbors_node = self.get_direct_neighbors(node_id)
                neighbors_not_in_category = [node for node in neighbors_node if
                                             node not in nodes_in_category]  # list(set(neighbors_node) & set(nodes_in_category))
                neighbors.update(neighbors_not_in_category)

        neighbors = list(neighbors)

        return nodes_in_category, color_category, neighbors

    def get_nodes_in_categories(self, group_id, categories_ids):
        nodes_in_categories = []

        if self.additional_parameters:
            for node_key, node_val in self.additional_parameters[
                "nodes"].items():
                if group_id in node_val["discrete_features"].keys():
                    node_categories = node_val["discrete_features"][group_id]
                    if any(x in categories_ids for x in node_categories):
                        nodes_in_categories.append(node_key)

        return nodes_in_categories

    def get_node_parameters_discrete(self, node_id):
        found = False
        for cpd in self.parameters:
            if cpd.variable == node_id:
                if len(cpd.variables) > 1:
                    list_vars = copy.deepcopy(cpd.variables)
                    list_vars.remove(node_id)
                    cpd.marginalize(list_vars)

                node_parameters_values = list(
                    np.round(cpd.get_values().flat, decimals=3))
                if cpd.state_names:
                    node_parameters_states = cpd.state_names[node_id]
                else:
                    node_parameters_states = []
                    for i, state in enumerate(node_parameters_values):
                        node_parameters_states.append("State " + str(i))
                found = True
                break

        if not found:
            raise Exception("Node parameters not found")

        return node_parameters_states, node_parameters_values

    def save_joint_dist(self, joint_dist, name_dist, only_marginals=False):
        inference_user_tmp_dir = os.path.join(settings.BN_INFERENCE_TMP_DIR)
        if name_dist.startswith("joint_dist_cond"):
            self.joint_dist_cond_path = os.path.join(inference_user_tmp_dir,
                                                     name_dist)
            joint_dist_path = self.joint_dist_cond_path
        elif name_dist.startswith("joint_dist"):
            self.joint_dist_path = os.path.join(inference_user_tmp_dir,
                                                name_dist)
            joint_dist_path = self.joint_dist_path
        else:
            raise Exception(
                f"Joint dist name must begin with joint_dist or joint_dist_cond. {name_dist} is not supported")

        if not os.path.exists(inference_user_tmp_dir):
            os.mkdir(inference_user_tmp_dir)

        sigma_marginals = joint_dist["sigma"].diagonal()

        np.save(joint_dist_path + "_mu", joint_dist["mu"])
        np.save(joint_dist_path + "_sigma_marginals", sigma_marginals)
        if not only_marginals:
            np.savez_compressed(joint_dist_path + "_sigma", allow_pickle=False,
                                sigma=joint_dist["sigma"])

    def load_joint_dist(self, path, only_marginals=False):
        joint_dist = {
            "mu": np.load(path + "_mu.npy")
        }
        if only_marginals:
            joint_dist["sigma"] = np.load(path + "_sigma_marginals.npy")
        else:
            joint_dist["sigma"] = np.load(path + "_sigma.npz")["sigma"]

        self.joint_dist = joint_dist

    def get_evidence(self):
        """
        Returns the known evidence.
        @return: A list of evidence elements.
        """
        return self.evidence

    def set_evidences(self, node_ids, evidence_value, evidence_scale,
                      joint_dist_cond_name="joint_dist_cond"):
        if self.data_type == "continuous":
            if evidence_scale == "scalar":
                self.evidence.set(node_ids, evidence_value)
            elif evidence_scale == "num_std_deviations":
                for node_id in node_ids:
                    inference_mode = True
                    evidence_value_not_set = None
                    mean, std_dev = self.get_marginal_mean_std_dev(
                        evidence_value_not_set, inference_mode, node_id,
                        evidences=self.evidence,
                        joint_dist_path=self.joint_dist_path,
                        joint_dist_cond_path=self.joint_dist_cond_path)
                    evidence_value_node = mean + std_dev * evidence_value
                    self.evidence.set([node_id], evidence_value_node)
                    nodes_names = list(self.graph.nodes())
                    self.load_joint_dist(self.joint_dist_path)
                    joint_dist_cond, not_evidences_nodes_order = GaussianExact.condition_on_evidence(
                        self.joint_dist, nodes_names, self.evidence)
                    self.evidence.set_not_evidence_nodes_order(
                        not_evidences_nodes_order)
                    self.save_joint_dist(joint_dist_cond, joint_dist_cond_name,
                                         only_marginals=False)
            else:
                raise Exception(
                    f"Evidence scale {evidence_scale} is not supported.")
            nodes_names = list(self.graph.nodes())
            self.load_joint_dist(self.joint_dist_path)
            joint_dist_cond, not_evidences_nodes_order = GaussianExact.condition_on_evidence(
                self.joint_dist, nodes_names, self.evidence)
            self.evidence.set_not_evidence_nodes_order(
                not_evidences_nodes_order)
            self.save_joint_dist(joint_dist_cond, joint_dist_cond_name,
                                 only_marginals=False)
        else:
            raise Exception("Discrete case not supported yet")

    def clear_evidences(self, nodes_ids,
                        joint_dist_cond_name="joint_dist_cond"):
        self.evidence.clear(nodes_ids)
        nodes_names = list(self.graph.nodes())

        if self.data_type == "continuous":
            if self.evidence.is_empty():
                self.evidence.set_not_evidence_nodes_order(None)
            else:
                joint_dist = self.load_joint_dist(self.joint_dist_path)
                joint_dist_cond, not_evidences_nodes_order = GaussianExact.condition_on_evidence(
                    joint_dist, nodes_names, self.evidence)
                self.evidence.set_not_evidence_nodes_order(
                    not_evidences_nodes_order)
                self.save_joint_dist(joint_dist_cond, joint_dist_cond_name,
                                     only_marginals=False)
        else:
            raise Exception("Discrete case not supported yet")

    def get_marginal_mean_std_dev(self, evidence_value=None,
                                  inference_mode=None, node_id=None,
                                  group_categories=None,
                                  evidences=None, joint_dist_path=None,
                                  joint_dist_cond_path=None):
        if inference_mode is None and evidence_value is None and \
                node_id is None and group_categories is None:
            start_joint_dist = self.load_joint_dist(joint_dist_path,
                                                    only_marginals=True)
            current_joint_dist = self.load_joint_dist(joint_dist_cond_path,
                                                      only_marginals=True)

            all_nodes = list(self.graph.nodes())
            not_evidences_nodes_order = evidences.get_not_evidence_nodes_order()

            start_means, start_std_devs = GaussianExact.marginal(
                start_joint_dist, all_nodes,
                marginal_nodes_ids=not_evidences_nodes_order,
                multivariate=False)

            current_means, current_std_devs = GaussianExact.marginal(
                current_joint_dist, not_evidences_nodes_order,
                marginal_nodes_ids=not_evidences_nodes_order,
                multivariate=False)

            mean = [start_means, current_means]
            std_dev = [start_std_devs, current_std_devs]
        elif group_categories is not None:
            start_means = []
            current_means = []
            start_std_devs = []
            current_std_devs = []

            for marginal_nodes_names in group_categories:
                start_joint_dist = self.load_joint_dist(joint_dist_path,
                                                        only_marginals=False)
                current_joint_dist = self.load_joint_dist(joint_dist_cond_path,
                                                          only_marginals=False)
                all_nodes = list(self.graph.nodes())
                not_evidences_nodes_order = evidences.get_not_evidence_nodes_order()

                start_means_category, start_std_devs_category = GaussianExact.marginal(
                    start_joint_dist, all_nodes,
                    marginal_nodes_ids=marginal_nodes_names,
                    multivariate=True)

                current_means_category, current_std_devs_category = GaussianExact.marginal(
                    current_joint_dist,
                    not_evidences_nodes_order,
                    marginal_nodes_ids=marginal_nodes_names,
                    multivariate=True)

                start_means.append(start_means_category)
                current_means.append(current_means_category)
                start_std_devs.append(start_std_devs_category)
                current_std_devs.append(current_std_devs_category)

            mean = [start_means, current_means]
            std_dev = [start_std_devs, current_std_devs]
        else:
            if inference_mode and evidence_value is None and not evidences.is_empty() and joint_dist_cond_path:
                joint_dist = self.load_joint_dist(joint_dist_cond_path,
                                                  only_marginals=True)
                not_evidences_nodes_order = evidences.get_not_evidence_nodes_order()
            else:
                joint_dist = self.load_joint_dist(joint_dist_path,
                                                  only_marginals=True)
                not_evidences_nodes_order = list(self.graph.nodes())
            marginals_nodes_names = [node_id]
            mean, std_dev = GaussianExact.marginal(joint_dist,
                                                   not_evidences_nodes_order,
                                                   marginals_nodes_names,
                                                   only_one_marginal=True)

        return mean, std_dev

    def get_node_parameters_continuous(self, node_id):
        evidence_value = self.evidence.get(node_id)
        mean, std_dev = self.get_marginal_mean_std_dev(evidence_value, True,
                                                       node_id,
                                                       evidences=self.evidence,
                                                       joint_dist_path=self.joint_dist_path,
                                                       joint_dist_cond_path=self.joint_dist_cond_path)

        init_mean, init_std_dev = self.get_marginal_mean_std_dev(
            evidence_value, False, node_id, evidences=self.evidence,
            joint_dist_path=self.joint_dist_path,
            joint_dist_cond_path=self.joint_dist_cond_path)
        if init_mean != mean and init_std_dev != std_dev:
            gaussian_pdf_plot = density_functions_bn(
                [init_mean, mean], [init_std_dev, std_dev], evidence_value)
        else:
            gaussian_pdf_plot = density_functions_bn(mean,
                                                     std_dev,
                                                     evidence_value)

        return gaussian_pdf_plot, evidence_value

    def get_probabilities_effect(self, group_categories=None):
        means, std_devs = self.get_marginal_mean_std_dev(
            group_categories=group_categories, evidences=self.evidence,
            joint_dist_path=self.joint_dist_path,
            joint_dist_cond_path=self.joint_dist_cond_path)
        start_means, start_std_devs = means[0], std_devs[0]
        current_means, current_std_devs = means[1], std_devs[1]

        return start_means, start_std_devs, current_means, current_std_devs

    def get_node_is_evidence_set(self, node_id):
        if self.evidence.get(node_id) is not None:
            return True
        else:
            return False

    def get_node_connections_info(self, node_id):
        parents = list(self.graph.predecessors(node_id))
        children = list(self.graph.successors(node_id))
        neighbors = parents + children

        num_parents = len(parents)
        num_children = len(children)
        num_neighbors = len(neighbors)

        top_parents = []
        if num_parents > 0:
            num_top_parents = min(3, num_parents)
            for parent in parents:
                edge_data = self.graph.get_edge_data(parent, node_id)
                top_parents.append((parent, edge_data["weight"]))
            top_parents.sort(key=itemgetter(1), reverse=True)
            top_parents = top_parents[0:num_top_parents]

        top_children = []
        if num_children > 0:
            num_top_children = min(3, num_children)
            for child in children:
                edge_data = self.graph.get_edge_data(node_id, child)
                top_children.append((child, edge_data["weight"]))
            top_children.sort(key=itemgetter(1), reverse=True)
            top_children = top_children[0:num_top_children]

        return num_parents, num_children, num_neighbors, top_parents, top_children

    def get_edge_info(self, source_node, target_node):
        weight = 1
        for (x, y, edge_data) in self.graph.edges(data=True):
            if x == source_node and y == target_node:
                weight = edge_data["weight"]
                break

        return weight

    def get_evidences_names(self):
        return self.evidence.get_names()

    # TODO: merge is_deseparated and get_reachable_nodes implementations
    def is_dseparated(self, start_nodes, observed_nodes, end_nodes):
        """
        Checks if two sets of nodes (start_nodes and end_nodes) are d-separated
            given another one (observed_nodes).
        @param start_nodes
        @param observed_nodes:
        @param end_nodes:
        @return:
        """
        # Following Reachable algorithm in Koller and Friedman (2009),
        # "Probabilistic Graphical Models: Principles and Techniques", page 75.

        # Phase I: Insert ancestors of observations into a list
        visit_nodes = observed_nodes.copy()
        obs_ancestors = set()

        while len(visit_nodes) > 0:
            node_id = visit_nodes.pop()
            node_parents = list(self.graph.predecessors(node_id))
            visit_nodes += list(set(node_parents) - set(observed_nodes))
            obs_ancestors.update(node_id)

        # Phase II: traverse active trails starting from start_nodes
        start_node_count = 0
        independent = True
        while len(start_nodes) > start_node_count and independent:
            via_nodes = [(start_nodes[start_node_count], "up")]
            visited = set()

            while len(via_nodes) > 0:
                node_name, direction = via_nodes.pop()
                if (node_name, direction) not in visited:
                    visited.add((node_name, direction))

                    if node_name not in observed_nodes and \
                            node_name in end_nodes:
                        independent = False

                    if direction == "up" and node_name not in observed_nodes:
                        node_parents = list(self.graph.predecessors(node_name))
                        node_children = list(self.graph.successors(node_name))
                        for parent_node in node_parents:
                            via_nodes.append((parent_node, "up"))
                        for child_node in node_children:
                            via_nodes.append((child_node, "down"))
                    elif direction == "down":
                        if node_name not in observed_nodes:
                            node_children = list(
                                self.graph.successors(node_name))
                            for child_node in node_children:
                                via_nodes.append((child_node, "down"))
                        if node_name in obs_ancestors:
                            node_parents = list(
                                self.graph.predecessors(node_name))
                            for parent_node in node_parents:
                                via_nodes.append((parent_node, "up"))
            start_node_count += 1
        return independent

    def get_reachable_nodes(self, start_nodes):
        """
        Returns the nodes reachable from start_nodes given the evidence via
            active trails. It follows Koller and Friedman (2009),
            "Probabilistic Graphical Models: Principles and Techniques",
            page 75.
        @param start_nodes:
        @return: A list of reachable nodes.
        """
        # Phase I: Insert ancestors of observations on a list
        observed = self.evidence.get_names()
        visit_nodes = observed.copy()
        obs_ancestors = set()
        reachable_nodes = set()

        while len(visit_nodes) > 0:
            node_id = visit_nodes.pop()
            node_parents = list(self.graph.predecessors(node_id))
            visit_nodes += list(set(node_parents) - set(observed))
            obs_ancestors.update(node_id)

        # Phase II: traverse active trails starting from X
        start_node_count = 0
        while len(start_nodes) > start_node_count:
            via_nodes = [(start_nodes[start_node_count], "up")]
            visited = set()

            while len(via_nodes) > 0:
                (node_name, direction) = via_nodes.pop()
                if (node_name, direction) not in visited:
                    visited.add((node_name, direction))

                    if node_name not in observed:
                        reachable_nodes.update(node_name)

                    if direction == "up" and node_name not in observed:
                        node_parents = list(self.graph.predecessors(node_name))
                        node_children = list(self.graph.successors(node_name))
                        for parent_node in node_parents:
                            via_nodes.append((parent_node, "up"))
                        for child_node in node_children:
                            via_nodes.append((child_node, "down"))
                    elif direction == "down":
                        if node_name not in observed:
                            node_children = list(
                                self.graph.successors(node_name))
                            for child_node in node_children:
                                via_nodes.append((child_node, "down"))
                        if node_name in obs_ancestors:
                            node_parents = list(
                                self.graph.predecessors(node_name))
                            for parent_node in node_parents:
                                via_nodes.append((parent_node, "up"))
            start_node_count += 1
        return reachable_nodes

    def get_edges_between_nodes(self, nodes_ids):
        nodes_selection = set()

        for node_id in nodes_ids:
            node_children = self.get_children(node_id)
            for node_id_searched in nodes_ids:
                if node_id_searched in node_children:
                    nodes_selection.add(node_id)
                    nodes_selection.add(node_id_searched)

        nodes_selection = list(nodes_selection)

        return nodes_selection

    def filter_edges_by_weight(self, weights_range):
        weights_range = [float(num) for num in weights_range]

        self.graph = self.original_graph.copy()

        for (x, y, edge_data) in self.original_graph.edges(data=True):
            edge_weight = round(edge_data["weight"], 3)
            if edge_weight < weights_range[0] or edge_weight > weights_range[
                1]:
                self.graph.remove_edge(x, y)

    # @staticmethod
    # def clean_inference_tmp_user_folder(session_key):
    #     inference_user_tmp_dir = os.path.join(settings.BN_INFERENCE_TMP_DIR,
    #                                           session_key)
    #
    #     try:
    #         shutil.rmtree(inference_user_tmp_dir)
    #         print("Delete session: inference tmp user data deleted from disk")
    #     except Exception as _:
    #         print(
    #             "Delete session: no inference tmp user data available to delete")
    #         return 1
    #
    #     return 0

    # TODO: add default values for arguments
    # TODO: add estimation algorithm parameters
    def fit(self, df, algorithm, estimation, *, alpha=None, tol=None,
            max_iter=None, max_parents=None, features_classes=None,
            max_adjacency_size=None, est_param=None):
        """
        Builds a Bayesian network using the input data.
        @param df: Data set used to learn the Bayesian network.
        @param algorithm: Algorithm used to learn the structure of the network.
        @param estimation: Estimation type used to learn the parameters of the
            network.
        @param features_classes:
        @param max_adjacency_size:
        @param max_parents:
        @param max_iter:
        @param tol:
        @param alpha:
        """

        data_type = dataframe_get_type(df)

        # Alternative way (easier, but requires the same parameters for all):
        # structure_learning = globals()[algorithm]
        # self.graph = structure_learning(df, algorithm, estimation).run()
        # Another option is to have dictionaries by constructor parameters

        if algorithm == 'CL':
            self.graph = CL(df, data_type=data_type).run()
        elif algorithm == 'FastIamb':
            self.graph = FastIamb(df, data_type=data_type,
                                  alpha=alpha).run()
        elif algorithm == 'Glasso':
            self.graph = Glasso(df, data_type=data_type, alpha=alpha,
                                tol=tol, max_iter=max_iter).run()
        elif algorithm == 'Gs':
            self.graph = Gs(df, data_type=data_type,
                            alpha=alpha).run()
        elif algorithm == 'Hc':
            self.graph = Hc(df, data_type=data_type,
                            max_number_parents=max_parents,
                            iterations=max_iter).run()
        elif algorithm == 'HcTabu':
            self.graph = HcTabu(df, data_type=data_type,
                                max_number_parents=max_parents,
                                iterations=max_iter).run()
        elif algorithm == 'HitonPC':
            self.graph = HitonPC(df, data_type=data_type).run()
        elif algorithm == 'Iamb':
            self.graph = Iamb(df, data_type=data_type,
                              alpha=alpha).run()
        elif algorithm == 'InterIamb':
            self.graph = InterIamb(df, data_type=data_type,
                                   alpha=alpha).run()
        elif algorithm == 'Lr':
            self.graph = Lr(df, data_type=data_type).run()
        elif algorithm == 'MBC':
            self.graph = MBC(df, data_type=data_type,
                             features_classes=features_classes).run()
        elif algorithm == 'MiContinuous':
            self.graph = MiContinuous(df, data_type=data_type).run()
        elif algorithm == 'MMHC':
            self.graph = MMHC(df, data_type=data_type).run()
        elif algorithm == 'MMPC':
            self.graph = MMPC(df, data_type=data_type).run()
        elif algorithm == 'NB':
            self.graph = NB(df, data_type=data_type,
                            features_classes=features_classes).run()
        elif algorithm == 'PC':
            self.graph = PC(df, data_type=data_type,
                            max_number_parents=max_parents,
                            max_adjacency_size=max_adjacency_size,
                            alpha=alpha).run()
        elif algorithm == 'Pearson':
            self.graph = Pearson(df, data_type=data_type).run()
        elif algorithm == 'SparseBn':
            self.graph = SparseBn(df, data_type=data_type).run()
        elif algorithm == 'Tan':
            self.graph = Tan(df, data_type=data_type,
                             features_classes=features_classes).run()

        if _check_estimator(estimation):
            params_learning = globals()[estimation]
            self.parameters = params_learning(df,
                                              graph=self.graph,
                                              algorithm_parameters=est_param,
                                              data_type=data_type).run()

        if data_type == 'continuous':
            self._set_topological_order()
            self.joint_dist = GaussianExact.joint(self.topological_order,
                                                  list(self.graph.nodes()),
                                                  self.parameters)
