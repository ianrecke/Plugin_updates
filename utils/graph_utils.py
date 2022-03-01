import os
import numpy as np
import numba
import networkx as nx
from operator import itemgetter
import bn_utils
import pandas as pd
import math
import plotly
import plotly.graph_objs as plotly_graph
import queue
from profilehooks import timecall


################# Getters ###############################
@numba.jit(nopython=True)
def adjs_by_axis_a(matrix, x):
    matrix_filter = matrix[x, :] > 0
    adjs = np.where(matrix_filter)[0]

    return set(adjs)


@numba.jit(nopython=True)
def adjs_by_axis_b(matrix, x):
    matrix_filter = matrix[:, x] > 0
    adjs = np.where(matrix_filter)[0]

    return set(adjs)


@numba.jit(nopython=True)
def adjacencies(matrix, x):
    adjs_axis_a = adjs_by_axis_a(matrix, x)
    adjs_axis_b = adjs_by_axis_b(matrix, x)

    all_adjs = adjs_axis_a.union(adjs_axis_b)

    return all_adjs


@numba.jit(nopython=True)
def undirecteds(matrix, x):
    adjs_axis_a = adjs_by_axis_a(matrix, x)
    adjs_axis_b = adjs_by_axis_b(matrix, x)

    result = adjs_axis_b & adjs_axis_a

    return result


@numba.jit(nopython=True)
def children(matrix, x):
    adjs_axis_a = adjs_by_axis_a(matrix, x)
    adjs_axis_b = adjs_by_axis_b(matrix, x)

    result = adjs_axis_a - adjs_axis_b

    return result


@numba.jit(nopython=True)
def parents(matrix, x):
    adjs_axis_a = adjs_by_axis_a(matrix, x)
    adjs_axis_b = adjs_by_axis_b(matrix, x)

    result = adjs_axis_b - adjs_axis_a

    return result


def is_undirected(matrix, x, y):
    undirected = matrix[x, y] > 0 and matrix[y, x] > 0
    return undirected


def is_children(matrix, x, y):
    child = matrix[x, y] > 0 and matrix[y, x] == 0
    return child


def is_parent(matrix, x, y):
    parent = matrix[x, y] == 0 and matrix[y, x] > 0
    return parent


def is_connected(matrix, x, y):
    return is_parent(matrix, x, y) or is_children(matrix, x, y)


################# Setters ###############################


def add_children(matrix, x, ch, bic=1):
    matrix[x, ch] = bic

    return matrix


def remove_children(matrix, x, ch):
    if matrix[x, ch] == 0:
        raise KeyError

    matrix[x, ch] = 0

    return matrix


def add_parent(matrix, x, parent):
    matrix[parent, x] = 1

    return matrix


def remove_parent(matrix, x, parent):
    if matrix[parent, x] == 0:
        raise KeyError

    matrix[parent, x] = 0

    return matrix


def add_undirected(matrix, x, y):
    add_children(matrix, x, y)
    add_parent(matrix, x, y)

    return matrix


def remove_undirected(matrix, x, y):
    remove_children(matrix, x, y)
    remove_parent(matrix, x, y)

    return matrix


def traverse_semi_directed(g, x, y):
    """Returns y if there is a directed """
    if is_undirected(g, x, y) or is_children(g, x, y):
        return y
    return None


def exists_unblocked_semi_directed_path(g, origin, dest, cond_set, bound):
    """Checks if there exists a unblocked semi directed path (that is, there could be a possible path) from
    origin to dest, while conditioning on cond_set"""
    if bound == -1:
        bound = 1000

    q = queue.Queue()
    v = set()
    q.put(origin)
    v.add(origin)

    e = None
    distance = 0

    while not q.empty():
        t = q.get()
        if t == dest:
            return True

        if e == t:
            e = None
            distance += 1
            if distance > bound:
                return False

        for u in adjacencies(g, t):
            c = traverse_semi_directed(g, t, u)
            if c is None:
                continue

            if c in cond_set:
                continue

            if c == dest:
                return True

            if c not in v:
                v.add(c)
                q.put(c)

                if e is None:
                    e = c
    return False


@timecall(immediate=True)
# @FGES.profile_each_line
def union_graphs(graph_global, graph_local, local_global_indices):
    n = graph_local.shape[0]

    for node in range(n):
        adjs = children(graph_local, node)

        for adj in adjs:
            node_global = local_global_indices[node]
            adj_global = local_global_indices[adj]
            if graph_global[node_global, adj_global] == 0:
                cond_set = parents(graph_global, adj_global)
                valid = not exists_unblocked_semi_directed_path(graph_global,
                                                                adj_global,
                                                                node_global,
                                                                cond_set,
                                                                graph_global.shape[
                                                                    0])

                if valid:
                    add_children(graph_global, node_global, adj_global,
                                 graph_local[node, adj])
            else:
                graph_global[node_global, adj_global] += graph_local[node, adj]

    return graph_global


def intersect_graphs(graph_global, graph_local, local_global_indices):
    n = graph_local.shape[0]
    positive_edges = graph_local[graph_local > 0]
    minimum = positive_edges[min(3, len(positive_edges) - 1)] if len(
        positive_edges) else 0
    for node in range(n):
        for adj in range(n):
            node_global = local_global_indices[node]
            adj_global = local_global_indices[adj]
            if graph_global[node_global, adj_global] < 0:
                continue

            elif graph_global[node_global, adj_global] > 0:
                bic = graph_local[node, adj] if graph_local[
                                                    node, adj] > minimum else 0.0
                graph_global[node_global, adj_global] = np.min(
                    [bic, graph_global[node_global, adj_global]])


            elif graph_local[node, adj] > 0:
                add_children(graph_global, node_global, adj_global,
                             graph_local[node, adj])

            else:
                graph_global[node_global, adj_global] = -1

    graph_global[graph_global == -1] = 0

    return graph_global


def intersection_graphs_tops(graph_global, graph_local, local_global_indices,
                             only_remove_if_global_bad=False):
    graph_filter = graph_local > 0
    positive_edges_inds = np.argwhere(graph_filter).tolist()
    positive_edges = []
    for edge in positive_edges_inds:
        x = edge[0]
        y = edge[1]
        positive_edges.append(((x, y), graph_local[x, y]))
    positive_edges.sort(key=itemgetter(1), reverse=True)
    positive_bics_values = [bic[1] for bic in positive_edges]

    max_candidates = len(positive_bics_values)
    num_candidates = bn_utils.hypothesis_test_related_genes(max_candidates,
                                                            positive_bics_values)

    positive_edges_tops = positive_edges[0: num_candidates]
    positive_edges_worst = positive_edges[num_candidates:]

    for edge in positive_edges_tops:
        x, y = edge[0]
        bic = edge[1]

        x_global = local_global_indices[x]
        y_global = local_global_indices[y]
        if graph_global[x_global, y_global] > 0:
            graph_global[x_global, y_global] += 1

    for edge in positive_edges_worst:
        x, y = edge[0]
        bic = edge[1]

        x_global = local_global_indices[x]
        y_global = local_global_indices[y]

        if not only_remove_if_global_bad:
            if graph_global[x_global, y_global] > 0:
                remove_children(graph_global, x_global, y_global)
        else:
            adjs = children(graph_global, x_global)
            positive_edges_global = []
            for adj in adjs:
                positive_edges_global.append(
                    ((x_global, adj), graph_global[x_global, adj]))
            positive_edges_global.sort(key=itemgetter(1), reverse=True)
            positive_bics_values_global = [bic[1] for bic in
                                           positive_edges_global]

            max_candidates = len(positive_bics_values_global)
            num_candidates = bn_utils.hypothesis_test_related_genes(
                max_candidates, positive_bics_values_global)
            positive_edges_global_tops = positive_edges_global[
                                         0: num_candidates]
            positive_edges_global_worst = positive_edges_global[
                                          num_candidates:]
            global_bad = False
            for edge in positive_edges_global_worst:
                x, y = edge[0]
                if x == x_global and y == y_global:
                    global_bad = True
                    break

            if global_bad:
                remove_children(graph_global, x_global, y_global)

    return graph_global


def remove_not_related_nodes(graph_global):
    n = graph_global.shape[0]

    for node in range(n):
        adjs = children(graph_global, node)
        positive_edges = []
        for adj in adjs:
            positive_edges.append(((node, adj), graph_global[node, adj]))
        positive_edges.sort(key=itemgetter(1), reverse=True)
        positive_bics_values = [bic[1] for bic in positive_edges]

        max_candidates = len(positive_bics_values)
        num_candidates = bn_utils.hypothesis_test_related_genes(max_candidates,
                                                                positive_bics_values)

        positive_edges_tops = positive_edges[0: num_candidates]
        positive_edges_worst = positive_edges[num_candidates:]

        for edge in positive_edges_worst:
            x, y = edge[0]
            remove_children(graph_global, x, y)

        done = True

    return 0


def force_hubs_directions(graph, list_hubs):
    n = graph.shape[0]

    for node in range(n):
        node_parents = parents(graph, node)
        if node in list_hubs:
            for par in node_parents:
                if par not in list_hubs:
                    score = graph[par, node]
                    remove_parent(graph, node, par)

                    cond_set = parents(graph, par)
                    valid = not exists_unblocked_semi_directed_path(graph, par,
                                                                    node,
                                                                    cond_set,
                                                                    graph.shape[
                                                                        0])

                    if valid:
                        add_children(graph, node, par, score)
                    else:
                        pass
                        # add_parent(graph, node, par)

    return 0


def remove_edges_children_hubs(graph, list_hubs, n_parents=None):
    n = graph.shape[0]

    nx_graph = nx.from_numpy_array(graph).to_undirected()

    for node in range(n):
        # print("Computing node ", node)

        if node not in list_hubs:
            neighbors = parents(graph, node) | children(graph, node)

            hubs_of_node = []
            for hub in list_hubs:
                try:
                    path_node_hub = nx.shortest_path(nx_graph, node, hub)
                    # Path must not traverse through another hubs:
                    hubs_in_path = any(
                        True for i in path_node_hub[0:len(path_node_hub) - 1]
                        if i in list_hubs)
                    if not hubs_in_path:
                        hubs_of_node.append(hub)
                except nx.exception.NetworkXNoPath as e:
                    pass

            for brother in neighbors:  # Remove all connections between this child and all children
                if brother not in list_hubs:  # Brother is not hub
                    # Remove connections between brothers
                    if is_parent(graph, node, brother):
                        remove_parent(graph, node, brother)
                    elif is_children(graph, node, brother):
                        remove_children(graph, node, brother)

            neighbors_after = parents(graph, node) | children(graph, node)
            if len(neighbors_after) == 0:
                max_hubs_per_children = n_parents if n_parents else len(
                    hubs_of_node)
                i = 0
                while i < max_hubs_per_children and i < len(hubs_of_node):
                    add_children(graph, hubs_of_node[i], node)
                    i += 1
    return 0


def get_list_hubs(graph, percentile, method="degree", threshold_out_degree=2,
                  show_plots=False):
    if method == "degree":
        list_hubs = get_list_hubs_by_degree(graph, percentile, show_plots)
    elif method == "out_degree":
        list_hubs = get_list_hubs_by_out_degree(graph, percentile,
                                                threshold_out_degree,
                                                show_plots)
    elif method == "betweenness":
        list_hubs = get_list_hubs_by_betweenness(graph, percentile)
    elif method == "degree-betweenness":
        list_hubs_degree = get_list_hubs_by_degree(graph, percentile,
                                                   show_plots)
        list_hubs_betweenness = get_list_hubs_by_betweenness(graph, percentile)
        list_hubs = list(set(list_hubs_degree) & set(list_hubs_betweenness))
    else:
        raise Exception(
            "Method {} for list of hubs does not exist".format(method))

    print("-----------Num hubs detected: {}-----------".format(len(list_hubs)))

    return list_hubs


def get_list_hubs_by_betweenness(graph, percentile):
    nx_graph = nx.from_numpy_array(graph).to_undirected()

    betweenness_dict = nx.betweenness_centrality(
        nx_graph)  # Run betweenness centrality
    sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1),
                                reverse=True)
    hubs_by_betwenness_vals = np.array(
        [node[1] for node in sorted_betweenness], dtype=np.float64)
    percentile_rank = np.percentile(hubs_by_betwenness_vals, percentile)
    hubs_by_betwenness = [node[0] for node in sorted_betweenness if
                          node[1] >= percentile_rank]

    return hubs_by_betwenness


def get_list_hubs_by_degree(graph, percentile, show_plots=False):
    neighbors_counter = []
    n = graph.shape[0]

    for node in range(n):
        node_children = children(graph, node)
        node_parents = parents(graph, node)
        num_neighbors = len(node_children) + len(node_parents)
        neighbors_counter.append(num_neighbors)

    neighbors_counter = np.sort(np.array(neighbors_counter, dtype=np.int64))
    threshold_neighbors = math.ceil(
        np.percentile(neighbors_counter, percentile))

    if show_plots:
        plot_name = "Histogram num neighbors"
        trace_histogram = plotly_graph.Histogram(
            x=neighbors_counter,
            name=plot_name,
            autobinx=True,
        )
        layout = plotly_set_layout(title=plot_name,
                                   column_x_name="Num neighbors",
                                   column_y_name="Num nodes", all_x_labels=0)
        figure = plotly_graph.Figure(data=[trace_histogram], layout=layout)
        plotly.offline.plot(figure, filename='{}.html'.format(plot_name))

    list_hubs = []
    for node in range(n):
        node_children = children(graph, node)
        node_parents = parents(graph, node)
        num_neighbors = len(node_children) + len(node_parents)
        if num_neighbors >= threshold_neighbors:
            list_hubs.append(node)

    return list_hubs


def get_list_hubs_by_out_degree(graph, percentile=None,
                                threshold_out_degree=None, show_plots=False):
    children_counter = []
    n = graph.shape[0]

    if threshold_out_degree is None:
        for node in range(n):
            num_chilren = len(children(graph, node))
            children_counter.append(num_chilren)

        children_counter = np.sort(np.array(children_counter, dtype=np.int64))
        threshold_out_degree = math.ceil(
            np.percentile(children_counter, percentile))

    if show_plots:
        plot_name = "Histogram num children"
        trace_histogram = plotly_graph.Histogram(
            x=children_counter,
            name=plot_name,
            autobinx=True,
        )
        layout = plotly_set_layout(title=plot_name,
                                   column_x_name="Num children",
                                   column_y_name="Num nodes", all_x_labels=0)
        figure = plotly_graph.Figure(data=[trace_histogram], layout=layout)
        plotly.offline.plot(figure, filename='{}.html'.format(plot_name))

    list_hubs = []
    for node in range(n):
        num_children = len(children(graph, node))
        if num_children >= threshold_out_degree:
            print("Node {}; {}".format(node, num_children))
            list_hubs.append(node)

    return list_hubs


def get_nodes_by_num_parents(graph):
    n = graph.shape[0]

    nodes_by_num_parents = []
    for node in range(n):
        num_parents = len(parents(graph, node))
        nodes_by_num_parents.append((node, num_parents))

    nodes_by_num_parents = sorted(nodes_by_num_parents, key=lambda x: x[1],
                                  reverse=False)

    return nodes_by_num_parents


def get_root_nodes(graph):
    nodes_by_num_parents = get_nodes_by_num_parents(graph)
    roots = [node[0] for node in nodes_by_num_parents if node[1] == 0]

    return roots


def plotly_set_layout(title="", column_x_name="", column_y_name="",
                      all_x_labels=1, height=600):
    layout = plotly_graph.Layout(
        title=title,
        autosize=True,
        height=height,
        margin=plotly_graph.layout.Margin(
        ),
        xaxis=dict(
            title=column_x_name,
            automargin=True,
            dtick=all_x_labels,
        ),
        yaxis=dict(
            title=column_y_name,
            automargin=True,
        ),
    )

    return layout


"""Transform between adjacency matrix and list to compare with DREAM data"""


def adj_list_to_matrix(adj_list, node_names=None, probabilities=False):
    # Takes adj list as pandas, returns adj matrix np.matrix. Takes ordered list of names
    if node_names is None:
        node_names = list(
            set(adj_list.iloc[:, 0].values) | set(adj_list.iloc[:, 1].values))

    n = len(node_names)
    adj_matrix = np.zeros((n, n))

    for gene1, gene2, arc in adj_list.values:
        if arc:  # (not probabilities and arc) or (probabilities and arc >= 0.5):
            i = node_names.index(gene1)
            j = node_names.index(gene2)
            if probabilities:
                adj_matrix[i, j] = arc
            else:
                adj_matrix[i, j] = 1

    return adj_matrix, node_names


def matrix_to_adj_list(matrix, node_names):
    n = len(matrix)
    names1 = []
    names2 = []
    arcs = []
    for i in range(n):
        for j in range(n):
            names1.append(node_names[i])
            names2.append(node_names[j])
            arcs.append(str(int(matrix[i, j] > 0)))

    adj_list = {"Gene1": names1, "Gene2": names2, "Arc": arcs}
    adj_list = pd.DataFrame(adj_list)
    return adj_list


def undirect_all_edges(matrix):
    n = matrix.shape[0]
    for node in range(n):
        node_children = children(matrix, node)
        for child in node_children:
            if matrix[child, node] == 0:
                matrix[child, node] = matrix[node, child]


def list_hubs_adj_matrix_file(graph_file, percentile=96, method="out_degree",
                              threshold_out_degree=2):
    graph_file_name, graph_file_extension = os.path.splitext(graph_file)

    graph_pd = pd.read_csv(graph_file).iloc[:, 1:]
    n = graph_pd.shape[1]

    nodes_names_map = dict(
        zip(list(range(n)), graph_pd.columns.values.tolist()))
    graph = graph_pd.values
    list_hubs = get_list_hubs(graph, percentile, method, threshold_out_degree)

    nodes_names_hubs = [nodes_names_map[node_idx] for node_idx in list_hubs]
    graph_output_pd = pd.DataFrame(nodes_names_hubs, columns=["hub_node_name"])
    graph_output_pd.to_csv("{}_hubs.csv".format(graph_file_name))

    return list_hubs


def filter_arcs_by_threshold(graph, threshold):
    graph = graph > threshold
    graph = graph.astype(np.float64)

    return graph


def filter_threshold_adj_matrix_file(graph_file, threshold):
    graph_file_name, graph_file_extension = os.path.splitext(graph_file)

    graph_pd = pd.read_csv(graph_file).iloc[:, 1:]
    graph = graph_pd.values
    graph_output = filter_arcs_by_threshold(graph, threshold)
    graph_output_pd = pd.DataFrame(graph_output, columns=graph_pd.columns)
    graph_output_pd.to_csv(graph_file_name + "_{}.csv".format(threshold))

    return graph_output_pd


def filter_edges_by_threshold(graph, threshold=0):
    n = graph.shape[0]

    graph_filter = graph > 0
    positive_edges_inds = np.argwhere(graph_filter).tolist()
    positive_edges = []
    for edge in positive_edges_inds:
        x = edge[0]
        y = edge[1]
        positive_edges.append(((x, y), graph[x, y]))
    positive_edges.sort(key=itemgetter(1), reverse=True)
    num_edges = len(positive_edges)
    if threshold == 0:
        max_num_edges = num_edges
    else:
        max_num_edges = int(n * threshold)
    positive_edges_worst = positive_edges[min(max_num_edges, num_edges):]
    for edge in positive_edges_worst:
        x, y = edge[0]
        graph[x, y] = 0

    return graph


def txt_to_adj_matrix():
    base_path = "./local_graphs/DREAM tests/0 - Original Network_predictions DREAM/Challenge participants"

    dirs_methods = next(os.walk(base_path))[1]
    for dir_method in dirs_methods:
        dir_method = os.path.join(base_path, dir_method)
        for graph_file in os.listdir(dir_method):
            graph_file = os.path.join(dir_method, graph_file)
            graph_file_name, graph_file_extension = os.path.splitext(
                graph_file)

            if graph_file_extension == ".txt":
                print("Converting txt network to csv: ", graph_file_name)
                adj_list = np.loadtxt(graph_file, dtype=object)
                adj_list_pd = pd.DataFrame(adj_list,
                                           columns=['A', 'B', 'Probability'])
                adj_list_pd = adj_list_pd.astype({"Probability": np.float64})

                prob_thresholds = np.arange(0.5, 1, 0.1)
                # for threshold in prob_thresholds:
                adj_matrix, nodes_names = adj_list_to_matrix(adj_list_pd,
                                                             probabilities=True)
                adj_matrix_pd = pd.DataFrame(adj_matrix, columns=nodes_names)
                adj_matrix_pd.to_csv(graph_file_name + ".csv")
            """
            elif graph_file_extension == ".csv":
                os.remove(graph_file)
            """


################# Tests ###############################

if __name__ == '__main__':
    """
    n = 10
    matrix = np.random.randint(0, 2, size=(n, n)) #Random graph
    np.fill_diagonal(matrix, 0)

    x = 0
    adjs = adjacencies(matrix, x)
    unds = undirecteds(matrix, x)
    childs = children(matrix, x)
    pars = parents(matrix, x)

    y = 4
    add_children(matrix, x, y)
    remove_children(matrix, x, y)
    add_parent(matrix, x, y)
    remove_parent(matrix, x, y)
    add_undirected(matrix, x, y)
    remove_undirected(matrix, x, y)
    """
    # txt_to_adj_matrix()
    # filter_threshold_adj_matrix_file('./local_graphs/DREAM5_NetworkInference_Regression2_Network1.csv', threshold=0.85)
    list_hubs_true = list_hubs_adj_matrix_file('./local_graphs/True.csv',
                                               method="out_degree",
                                               threshold_out_degree=2)
    list_hubs_test = list_hubs_adj_matrix_file('./local_graphs/test.csv',
                                               method="out_degree",
                                               threshold_out_degree=2)
    hubs_ok = set(list_hubs_true) & set(list_hubs_test)

    print("TP hubs: ", hubs_ok)
    print("Num hubs TRUE: ", len(list_hubs_true))
    print("Num hubs test: ", len(list_hubs_test))
    print("Num TP hubs: ", len(hubs_ok))
    print("Num FP hubs: ", len(list_hubs_test) - len(hubs_ok))

    done = True
