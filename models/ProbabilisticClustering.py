import neurosuite_dj.helpers as global_helpers
from sklearn import covariance as sk_learn_cov
import numpy as np
import networkx
from apps.morpho_analyzer.helpers.stats import plotly_helpers

from .. import inference as bn_inference
import BayesianNetwork


class ProbabilisticClustering(BayesianNetwork):

    def __init__(self, name="BN", dataset=[], is_uploaded_bn_file=False,
                 graph=None, parameters=None,
                 model_original_import={}, features_classes=[], data_type=None,
                 session_id=None,
                 # ProbabilisticClustering params:
                 structures_data=None):

        super(ProbabilisticClustering, self).__init__(name, dataset,
                                                      is_uploaded_bn_file,
                                                      graph, parameters,
                                                      model_original_import,
                                                      features_classes,
                                                      data_type, session_id)

        self.structures_data = structures_data
        self.structures_ids = list(structures_data.keys())

        i = 0
        for structure_id, structure_data, in structures_data.items():
            self.weights_info = self.init_graph_edges_weights(
                structure_data["graph"])
            self.order_topological = self.set_topological_order(
                structure_data["graph"])

            if structure_data["joint_dist"] is not None:
                if self.data_type == "continuous":
                    joint_dist_name = "joint_dist_" + structure_id
                    self.set_nodes_parameters_continuous(
                        structure_data["joint_dist"], joint_dist_name)
                    # Set latest joint dists paths as the structure joint dists paths:
                    structure_data["joint_dist_path"] = self.joint_dist_path
                    structure_data[
                        "joint_dist_cond_path"] = self.joint_dist_cond_path
                else:
                    structure_data["joint_dist"] = structure_data["joint_dist"]

            structure_data["evidences"] = bn_inference.Evidence()
            structure_data["color"] = self.get_structure_color(i)
            i += 1

        # Set global joint dists path (structures union graph):
        self.joint_dist_path = ""
        self.joint_dist_cond_path = ""

        self.data_type = "continuous"
        self.parameters = parameters
        nodes_names = list(self.graph.nodes())

    def get_structure_color(self, structure_number):
        # https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
        # https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
        # https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib

        if structure_number < 10:
            structures_colors = {
                0: "#636363",
                1: "#6a3d9a",
                2: "#2ca02c",
                3: "#ff9896",
                4: "#8c564b",
                5: "#ff7f0e",
                6: "#ffffb3",
                7: "#1f77b4",
                8: "#fdae6b",
                9: "#e7298a",
            }
            color = structures_colors[structure_number]
        else:
            color = global_helpers.generate_random_color()

        return color

    def set_nodes_parameters_continuous(self, joint_dist, joint_dist_name):
        self.save_joint_dist(joint_dist, joint_dist_name)

    def get_node_parameters_continuous(self, node_id, inference_mode,
                                       structure_id=None):
        if structure_id is None:
            raise Exception("Cluster not provided")
        elif structure_id == "all":
            means = []
            std_devs = []
            mixture_weights = []
            structure_ids = []
            structure_colors = []
            for structure_id, structure_data, in self.structures_data.items():
                if 'joint_dist_path' in structure_data:
                    evidence_value, mean, std_dev = self.get_structure_marginal(
                        inference_mode, node_id, structure_id)
                    means.append(mean)
                    std_devs.append(std_dev)
                    mixture_weights.append(
                        structure_data["percentage_instances"])
                    structure_ids.append("Cluster " + str(structure_id))
                    structure_colors.append(structure_data["color"])

            gaussian_pdf_plot = plotly_helpers.density_functions_multi(means,
                                                                       std_devs,
                                                                       mixture_weights,
                                                                       evidence_value,
                                                                       structure_ids,
                                                                       structure_colors)
        else:
            evidence_value, mean, std_dev = self.get_structure_marginal(
                inference_mode, node_id, structure_id)

            init_mean, init_std_dev = self.get_init_structure_marginal(False,
                                                                       node_id,
                                                                       structure_id)

            gaussian_pdf_plot = plotly_helpers.density_functions_bn(
                [init_mean, mean], [init_std_dev, std_dev], evidence_value)

        return gaussian_pdf_plot, evidence_value

        #     gaussian_pdf_plot = plotly_helpers.density_functions_bn(mean, std_dev, evidence_value)
        #
        # return gaussian_pdf_plot, evidence_value

    def get_structure_marginal(self, inference_mode, node_id, structure_id,
                               scalar=True):
        structure_data = self.structures_data[structure_id]
        evidences = structure_data["evidences"]
        joint_dist_path = structure_data['joint_dist_path']
        joint_dist_cond_path = structure_data['joint_dist_cond_path']
        if scalar:
            evidence_value = evidences.get(node_id)
        else:
            evidence_value = None
        group_categories = None
        mean, std_dev = self.get_marginal_mean_std_dev(evidence_value,
                                                       inference_mode, node_id,
                                                       group_categories,
                                                       evidences,
                                                       joint_dist_path,
                                                       joint_dist_cond_path)
        return evidence_value, mean, std_dev

    def get_init_structure_marginal(self, inference_mode, node_id,
                                    structure_id, scalar=True):
        structure_data = self.structures_data[structure_id]
        evidences = structure_data["evidences"]
        joint_dist_path = structure_data['joint_dist_path']
        joint_dist_cond_path = structure_data['joint_dist_cond_path']
        if scalar:
            evidence_value = evidences.get(node_id)
        else:
            evidence_value = None
        group_categories = None

        init_mean, init_std_dev = self.get_marginal_mean_std_dev(
            evidence_value, inference_mode, node_id,
            evidences=evidences,
            joint_dist_path=joint_dist_path,
            joint_dist_cond_path=joint_dist_cond_path)

        # mean, std_dev = self.get_marginal_mean_std_dev(evidence_value, inference_mode, node_id, group_categories,
        #                                                evidences, joint_dist_path, joint_dist_cond_path)

        return init_mean, init_std_dev

    def set_evidences(self, node_ids, evidence_value, evidence_scale,
                      structure_id=None):
        if structure_id is None:
            raise Exception("Cluster not provided")
        elif structure_id == "all":
            for structure_id, structure_data, in self.structures_data.items():
                self.set_evidence_in_structure(node_ids, evidence_value,
                                               evidence_scale, structure_id)
        else:
            self.set_evidence_in_structure(node_ids, evidence_value,
                                           evidence_scale, structure_id)

    def set_evidence_in_structure(self, node_ids, evidence_value,
                                  evidence_scale, structure_id):
        structure_data = self.structures_data[structure_id]
        evidences = structure_data["evidences"]
        joint_dist_path = structure_data['joint_dist_path']
        joint_dist_cond_name = "joint_dist_cond_" + structure_id

        if evidence_scale == "scalar":
            evidences.set(node_ids, evidence_value)
            self.update_set_evidence_parameters(joint_dist_path, evidences,
                                                joint_dist_cond_name,
                                                structure_data)
        elif evidence_scale == "num_std_deviations":
            for node_id in node_ids:
                inference_mode = True
                scalar = False
                _, mean, std_dev = self.get_structure_marginal(inference_mode,
                                                               node_id,
                                                               structure_id,
                                                               scalar)

                evidence_value_node = mean + std_dev * evidence_value
                evidences.set([node_id], evidence_value_node)
                self.update_set_evidence_parameters(joint_dist_path, evidences,
                                                    joint_dist_cond_name,
                                                    structure_data)
        else:
            raise Exception(
                "Evidence scale {} is not supported".format(evidence_scale))

    def update_set_evidence_parameters(self, joint_dist_path, evidences,
                                       joint_dist_cond_name, structure_data):
        nodes_names = list(self.graph.nodes())
        joint_dist = self.load_joint_dist(joint_dist_path)
        joint_dist_cond, not_evidences_nodes_order = bn_inference.GaussianExact.condition_on_evidence(
            joint_dist, nodes_names, evidences)
        evidences.set_not_evidence_nodes_order(not_evidences_nodes_order)
        self.save_joint_dist(joint_dist_cond, joint_dist_cond_name,
                             only_marginals=False)
        structure_data["joint_dist_cond_path"] = self.joint_dist_cond_path

    def get_evidences_names(self, structure_id):
        nodes_evidences = []
        if structure_id is None:
            return nodes_evidences

        if structure_id == "all":
            for structure_id, structure_data in self.structures_data.items():
                nodes_evidences += structure_data["evidences"].get_names()
        elif structure_id != "common":
            nodes_evidences += self.structures_data[structure_id][
                "evidences"].get_names()

        return nodes_evidences

    def clear_evidences(self, nodes_ids, structure_id=None):
        if structure_id is None:
            raise Exception("Cluster not provided")
        elif structure_id == "all":
            for structure_id, structure_data, in self.structures_data.items():
                self.clear_evidence_in_structure(nodes_ids, structure_id)
        else:
            self.clear_evidence_in_structure(nodes_ids, structure_id)

    def clear_evidence_in_structure(self, nodes_ids, structure_id):
        structure_data = self.structures_data[structure_id]
        evidences = structure_data["evidences"]
        joint_dist_path = structure_data['joint_dist_path']
        joint_dist_cond_name = "joint_dist_cond_" + structure_id

        evidences.clear(nodes_ids)
        nodes_names = list(self.graph.nodes())

        if self.data_type == "continuous":
            if evidences.is_empty():
                evidences.set_not_evidence_nodes_order(None)
            else:
                joint_dist = self.load_joint_dist(joint_dist_path)
                joint_dist_cond, not_evidences_nodes_order = bn_inference.GaussianExact.condition_on_evidence(
                    joint_dist, nodes_names, evidences)
                evidences.set_not_evidence_nodes_order(
                    not_evidences_nodes_order)
                self.save_joint_dist(joint_dist_cond, joint_dist_cond_name,
                                     only_marginals=False)
        else:
            raise Exception("Discrete case not supported yet")

    def update_glasso_paramters(self, alpha, tol, max_iter):
        nodes_names = list(self.graph.nodes())
        edges = list(self.graph.edges())
        self.graph.remove_edges_from(edges)

        for structure_id, structure_data, in self.structures_data.items():
            cov_matrix = structure_data["joint_dist"]["sigma"]
            _, precision_matrix = sk_learn_cov.graphical_lasso(cov_matrix,
                                                               alpha=alpha,
                                                               tol=tol,
                                                               max_iter=max_iter)
            adj_matrix = np.array(precision_matrix)
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = np.triu(adj_matrix)
            structure_data["graph"] = self.nx_graph_from_adj_matrix(adj_matrix,
                                                                    nodes_names)
            self.add_new_multi_graph_edges_net_attr(structure_data["graph"],
                                                    structure_id)

        return 0

    def nx_graph_from_adj_matrix(self, adj_matrix, nodes_names):
        nx_graph = networkx.from_numpy_matrix(adj_matrix,
                                              create_using=networkx.DiGraph)

        mapping = {}
        for i, node in enumerate(nx_graph.nodes()):
            mapping[node] = nodes_names[i]

        nx_graph = networkx.relabel_nodes(nx_graph, mapping)

        return nx_graph

    def get_nodes_in_category(self, group_id, category_id, structure_id,
                              show_neighbors):
        result = {}
        nodes_in_category = []
        color_category = ""
        neighbors = set()
        if structure_id == "all":
            graph = self.graph
        else:
            graph = self.structures_data[structure_id]["graph"]

        if self.additional_parameters:
            color_category = \
                self.additional_parameters["discrete_features"][group_id][
                    category_id]["color"]

            for node_key, node_val in self.additional_parameters[
                "nodes"].items():
                if group_id in node_val["discrete_features"].keys():
                    node_groups = node_val["discrete_features"][group_id]
                    if category_id in node_groups:
                        nodes_in_category.append(node_key)

        if show_neighbors:
            for node_id in nodes_in_category:
                if graph.has_node(node_id):
                    node_parents = list(graph.predecessors(node_id))
                    node_children = list(graph.successors(node_id))
                    neighbors_node = node_parents + node_children
                    neighbors_not_in_category = [node for node in
                                                 neighbors_node if
                                                 node not in nodes_in_category]  # list(set(neighbors_node) & set(nodes_in_category))
                    neighbors.update(neighbors_not_in_category)
                else:
                    node_not_in = True

        neighbors = list(neighbors)

        return nodes_in_category, color_category, neighbors

    def get_edges_between_nodes(self, nodes_ids, structure_id):
        nodes_selection = set()

        if structure_id == "all":
            graph = self.graph
        else:
            graph = self.structures_data[structure_id]["graph"]
        graph = self.graph

        for node_id in nodes_ids:
            if graph.has_node(node_id):
                node_children = list(graph.successors(node_id))
                for node_id_searched in nodes_ids:
                    if (node_id_searched is not node_id) and (
                            node_id_searched in node_children):
                        nodes_selection.add(node_id)
                        nodes_selection.add(node_id_searched)

        nodes_selection = list(nodes_selection)

        return nodes_selection
