"""
Graph utilities module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import queue
from operator import itemgetter

import networkx as nx
import numba
import numpy as np

from .statistics import hypothesis_test_related_genes


@numba.jit(nopython=True)
def adjs_by_axis(matrix, x, axis=0):
    """
    Returns the adjacent elements of a node in a particular axis given an
    adjacency matrix.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    axis : {0, 1}, default=0
        The axis to look at in the adjacency matrix.

    Returns
    -------
    set
        A set with the parents or children of the node.
    """

    if axis == 0:
        matrix_filter = matrix[:, x] > 0
    else:
        matrix_filter = matrix[x, :] > 0
    # matrix_filter = np.take(matrix, x, axis=axis) > 0
    adjs = np.where(matrix_filter)[0]

    return set(adjs)


@numba.jit(nopython=True)
def adjacencies(matrix, x):
    """
    Returns all the adjacent elements of a node in a graph given its adjacency
    matrix.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    set
        A set with the adjacent nodes of the selected node in the graph.
    """

    adjs_axis_0 = adjs_by_axis(matrix, x, axis=0)
    adjs_axis_1 = adjs_by_axis(matrix, x, axis=1)

    all_adjs = adjs_axis_0 | adjs_axis_1

    return all_adjs


@numba.jit(nopython=True)
def undirected_neighbors(matrix, x):
    """
    Returns the undirected neighbors of a node in the graph. They are supposed
    to be represented in the adjacency matrix by positive values in both
    axis.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    set
        A set with the undirected neighbors of the selected node.
    """

    adjs_axis_0 = adjs_by_axis(matrix, x, axis=0)
    adjs_axis_1 = adjs_by_axis(matrix, x, axis=1)

    result = adjs_axis_0 & adjs_axis_1

    return result


def children(matrix, x):
    """
    Returns the children of a particular node in the graph. The children of a
    node are supposed to be those elements with a positive value in the
    0-axis that are not undirected.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    set
        A set with the children of the selected node.
    """
    return adjs_by_axis(matrix, x, axis=0) - adjs_by_axis(matrix, x, axis=1)


@numba.jit(nopython=True)
def parents(matrix, x):
    """
    Returns the parents of a particular node in the graph. The parents of a
    node are supposed to be those elements with a positive value in the
    1-axis that are not undirected.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    set
        A set with the parents of the selected node.
    """

    return adjs_by_axis(matrix, x, axis=0) - adjs_by_axis(matrix, x, axis=1)


def is_undirected(matrix, x, y):
    """
    Checks if the edge between two nodes is undirected.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    y : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    bool
        Whether the edge between the nodes with indices `x` and `y` is
        undirected.
    """

    return matrix[x, y] > 0 and matrix[y, x] > 0


def has_edge(matrix, parent, child):
    """
    Checks if there is a directed edge between two nodes.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    parent : int
        The index of the parent in the matrix.

    child : int
        The index of the child in the matrix.

    Returns
    -------
    bool
        Whether there is an edge `x`->`y` in `matrix` or not.
    """

    return matrix[parent, child] > 0 and matrix[child, parent] == 0


def add_edge(matrix, parent, child, value=1):
    """
    Adds a directed edge to an adjacency matrix.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    parent : int
        The index of the parent in the matrix.

    child : int
        The index of the child in the matrix.

    value : float, default=1
        The value assigned to the new edge.

    Returns
    -------
    numpy.array
        The adjacency matrix with the new edge added.
    """

    matrix[parent, child] = value

    return matrix


def remove_edge(matrix, parent, child):
    """
    Removes the directed edge between two nodes.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    parent : int
        The index of the parent in the matrix.

    child : int
        The index of the child in the matrix.

    Returns
    -------
    numpy.array
        The adjacency matrix with the edge removed.

    Raises
    ------
    ValueError
        If there is no edge between the nodes.
    """

    if matrix[parent, child] == 0:
        raise ValueError(f'{parent} is not {child}\'s parent.')

    matrix[parent, child] = 0

    return matrix


def add_undirected(matrix, x, y):
    """
    Adds an undirected edge between two nodes to an adjacency matrix.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    y : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    numpy.array
        The adjacency matrix with the undirected edge added.
    """

    matrix = add_edge(matrix, x, y)
    matrix = add_edge(matrix, y, x)

    return matrix


def remove_undirected(matrix, x, y):
    """
    Removes the undirected edge between two nodes.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    y : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    numpy.array
        The adjacency matrix with the undirected edge removed.

    Raises
    ------
    ValueError
        If there is no undirected edge between the nodes.
    """

    try:
        matrix = remove_edge(matrix, x, y)
        matrix = remove_edge(matrix, y, x)
    except ValueError as e:
        raise e

    return matrix


def traverse_sd(matrix, x):
    """
    Returns elements for which there is an edge from the selected node, i.e.,
    all y that are present in an edge of the form x->y or x-y.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    x : int
        The index of a node of the graph in the matrix.

    Returns
    -------
    set
        A set with the nodes that are present in a directed or undirected
        edge starting from the selected node.
    """

    return adjs_by_axis(matrix, x, axis=0)


def exists_sd_path(matrix, origin, dest, cond_set, bound=1000):
    """
    Checks if there exists a semi directed path (that is, there could be a
    possible path) from origin to dest, while conditioning on a set of
    nodes that cannot be part of the path.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    origin : int
        The index of the origin node in the matrix.

    dest : int
        The index of the destination node in the matrix.

    cond_set :
        The indices of the nodes that cannot be part of the path.

    bound : int, default=1000
        The maximum length for the path.

    Returns
    -------
    bool
        Whether there is such a path or not.
    """

    q = queue.Queue()
    visited = set()
    q.put(origin)
    visited.add(origin)

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

        for c in traverse_sd(matrix, t):
            if c is None:
                continue

            if c in cond_set:
                continue

            if c == dest:
                return True

            if c not in visited:
                visited.add(c)
                q.put(c)

                if e is None:
                    e = c
    return False


def union(global_matrix, local_matrix, local2global):
    """Merges the local graph into the global graph.

    Parameters
    ----------
    global_matrix : numpy.array
        Adjacency matrix of the global graph.

    local_matrix : numpy.array
        Adjacency matrix of the local graph.

    local2global : dict
        Translation of the local matrix nodes' indices to their indices in the
        global one.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    n = local_matrix.shape[0]

    for node in range(n):
        ch = children(local_matrix, node)

        for child in ch:
            g_node = local2global[node]
            g_child = local2global[child]
            if global_matrix[g_node, g_child] == 0:
                cond_set = parents(global_matrix, g_child)
                if not exists_sd_path(global_matrix, g_child, g_node, cond_set,
                                      global_matrix.shape[0]):
                    global_matrix = add_edge(global_matrix, g_node, g_child,
                                             local_matrix[node, child])
            else:
                global_matrix[g_node, g_child] += local_matrix[node, child]

    return global_matrix


def intersect(global_matrix, local_matrix, local2global):
    """

    Parameters
    ----------
    global_matrix : numpy.array
        Adjacency matrix of the global graph.

    local_matrix : numpy.array
        Adjacency matrix of the local graph.

    local2global : dict
        Translation of the local matrix nodes' indices to their indices in the
        global one.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    n = local_matrix.shape[0]
    positive_edges = local_matrix[local_matrix > 0]
    minimum = positive_edges[min(3, len(positive_edges) - 1)] if len(
        positive_edges) else 0
    for node in range(n):
        for adj in range(n):
            node_global = local2global[node]
            adj_global = local2global[adj]
            if global_matrix[node_global, adj_global] < 0:
                continue

            elif global_matrix[node_global, adj_global] > 0:
                value = local_matrix[node, adj] if local_matrix[
                                                       node, adj] > minimum else 0.0
                global_matrix[node_global, adj_global] = np.min(
                    [value, global_matrix[node_global, adj_global]])

            elif local_matrix[node, adj] > 0:
                global_matrix = add_edge(global_matrix, node_global,
                                         adj_global, local_matrix[node, adj])

            else:
                global_matrix[node_global, adj_global] = -1

    global_matrix[global_matrix == -1] = 0

    return global_matrix


def remove_local_unrelated(global_matrix, local_matrix, local2global,
                           check_global=False):
    """
    Removes from the global graph these edges that are unrelated in the
    local one according to a hypothesis test.

    Parameters
    ----------
    global_matrix : numpy.array
        Adjacency matrix of the global graph.

    local_matrix : numpy.array
        Adjacency matrix of the local graph.

    local2global : dict
        Translation of the local matrix nodes' indices to their indices in the
        global one.

    check_global : bool, default=False
        Whether to check if the edges are unrelated in the global graph too
        before removing them or not.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    graph_filter = local_matrix > 0
    positive_edges_inds = np.argwhere(graph_filter).tolist()
    positive_edges = []
    for edge in positive_edges_inds:
        x = edge[0]
        y = edge[1]
        positive_edges.append(((x, y), local_matrix[x, y]))
    positive_edges.sort(key=itemgetter(1), reverse=True)
    values_local = [value[1] for value in positive_edges]

    max_candidates = len(values_local)
    num_candidates = hypothesis_test_related_genes(max_candidates,
                                                   values_local)

    best = positive_edges[: num_candidates]
    worst = positive_edges[num_candidates:]

    for edge in best:
        x, y = edge[0]

        x_global = local2global[x]
        y_global = local2global[y]
        if global_matrix[x_global, y_global] > 0:
            global_matrix[x_global, y_global] += 1

    for edge in worst:
        x, y = edge[0]

        x_global = local2global[x]
        y_global = local2global[y]

        if not check_global:
            if global_matrix[x_global, y_global] > 0:
                global_matrix = remove_edge(global_matrix, x_global, y_global)
        else:
            ch = children(global_matrix, x_global)
            positive_edges_global = []
            for child in ch:
                positive_edges_global.append(
                    ((x_global, child), global_matrix[x_global, child]))
            positive_edges_global.sort(key=itemgetter(1), reverse=True)
            values_global = [value[1] for value in positive_edges_global]

            max_candidates = len(values_global)
            num_candidates = hypothesis_test_related_genes(max_candidates,
                                                           values_global)
            global_worst = positive_edges_global[num_candidates:]
            global_bad = False
            for bad_edge in global_worst:
                x, y = bad_edge[0]
                if x == x_global and y == y_global:
                    global_bad = True
                    break

            if global_bad:
                global_matrix = remove_edge(global_matrix, x_global, y_global)

    return global_matrix


def remove_unrelated(matrix):
    """
    Removes, for each node, all the edges with unrelated nodes according to
    a hypothesis test.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    n = matrix.shape[0]

    for node in range(n):
        ch = children(matrix, node)
        positive_edges = []
        for child in ch:
            positive_edges.append(((node, child), matrix[node, child]))
        positive_edges.sort(key=itemgetter(1), reverse=True)
        values = [value[1] for value in positive_edges]

        max_candidates = len(values)
        num_candidates = hypothesis_test_related_genes(max_candidates, values)

        positive_edges_worst = positive_edges[num_candidates:]

        for edge in positive_edges_worst:
            x, y = edge[0]
            matrix = remove_edge(matrix, x, y)

    return matrix


def force_directions(matrix, hubs):
    """
    Remove edges between hubs parents and hubs if there is an unblocked
    semi-directed path between them that does not include the parent's
    parents.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    hubs : list
        The set of hubs to check.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    for node in hubs:
        node_parents = parents(matrix, node)
        for parent in node_parents:
            if parent not in hubs:
                score = matrix[parent, node]
                matrix = remove_edge(matrix, parent, node)

                cond_set = parents(matrix, parent)

                if not exists_sd_path(matrix, parent, node, cond_set,
                                      matrix.shape[0]):
                    matrix = add_edge(matrix, node, parent, score)

    return matrix


def remove_children_edges(matrix, hubs, n_parents=None):
    """
    Removes the edges between non-hubs and include posible edges if they
    become isolated.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    hubs : list
        The set of hubs to check.

    n_parents : int, optional

    """

    n = matrix.shape[0]

    nx_graph = nx.from_numpy_array(matrix).to_undirected()

    for node in range(n):
        if node not in hubs:
            neighbors = undirected_neighbors(matrix, node)

            node_hubs = []
            for hub in hubs:
                try:
                    path = nx.shortest_path(nx_graph, node, hub)
                    # Path must not traverse through other hubs:
                    hubs_in_path = any(i in hubs for i in path[:len(path) - 1])
                    if not hubs_in_path:
                        node_hubs.append(hub)
                except nx.exception.NetworkXNoPath:
                    pass

            # Remove all connections between this child and all children
            for brother in neighbors:
                if brother not in hubs:  # Brother is not hub
                    # Remove connections between brothers
                    if has_edge(matrix, brother, node):
                        matrix = remove_edge(matrix, brother, node)
                    elif has_edge(matrix, node, brother):
                        matrix = remove_edge(matrix, node, brother)

            neighbors_after = undirected_neighbors(matrix, node)
            if not neighbors_after:
                max_hubs = n_parents if n_parents is not None else len(
                    node_hubs)
                for i in range(min(max_hubs, len(node_hubs))):
                    matrix = add_edge(matrix, node_hubs[i], node)

    return matrix


def get_hubs(matrix, percentile=None, method='degree', threshold=None):
    """
    Returns all the nodes after some percentile according to some criteria.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    percentile : float, optional
        The percentile required to be part of the set of hubs.

    method : str, default='degree'
        The method used to get the set of hubs.

    threshold : int, optional
        If the selected method is 'out_degree', it represents the minimum
        amount of children needed for the hubs set. The function does not take
        into account the `percentile` if it is set.

    Returns
    -------
    list
        The list of hubs according to the selected method.

    Raises
    ------
    ValueError
        If the selected method is not supported.
    """

    if method == 'degree':
        hubs = _hubs_by_degree(matrix, percentile)
    elif method == 'out_degree':
        hubs = _hubs_by_out_degree(matrix, percentile, threshold)
    elif method == 'betweenness':
        hubs = _hubs_by_betweenness(matrix, percentile)
    elif method == 'degree-betweenness':
        list_hubs_degree = _hubs_by_degree(matrix, percentile)
        list_hubs_betweenness = _hubs_by_betweenness(matrix, percentile)
        hubs = list(set(list_hubs_degree) & set(list_hubs_betweenness))
    else:
        raise ValueError(f'Method {method} for list of hubs does not exist')

    return hubs


def _hubs_by_betweenness(matrix, percentile):
    """
    Returns all the nodes after some percentile according to their betweenness
    centrality values.
    """

    nx_graph = nx.from_numpy_array(matrix).to_undirected()

    betweenness = nx.betweenness_centrality(nx_graph)
    sorted_betweenness = sorted(betweenness.items(), key=itemgetter(1),
                                reverse=True)

    hubs_vals = np.array([node[1] for node in sorted_betweenness],
                         dtype=np.float64)
    threshold = np.percentile(hubs_vals, percentile)
    return [node[0] for node in sorted_betweenness if node[1] >= threshold]


def _hubs_by_degree(matrix, percentile):
    """
    Returns all the nodes after some percentile according to their degrees.
    """

    n = matrix.shape[0]
    neighbors_counter = [len(adjacencies(matrix, node)) for node in range(n)]

    sorted_counter = np.sort(np.array(neighbors_counter, dtype=np.int64))
    threshold = np.percentile(sorted_counter, percentile)

    return [node for node in range(n) if neighbors_counter[node] >= threshold]


def _hubs_by_out_degree(matrix, percentile=None, threshold=None):
    """
    Returns all the nodes after some percentile according to their out degrees.
    """

    n = matrix.shape[0]
    children_counter = [children(matrix, node) for node in range(n)]

    if threshold is None:
        sorted_counter = np.sort(np.array(children_counter, dtype=np.int64))
        threshold = np.percentile(sorted_counter, percentile)

    return [node for node in range(n) if children_counter[node] >= threshold]


def undirect(matrix):
    """
    Make all edges in the matrix undirected.

    Parameters
    ----------
    matrix : numpy.array
        Adjacency matrix that represents a graph.

    Returns
    -------
    numpy.array
        The resulting adjacency matrix.
    """

    # TODO: check if undirected edges can have different scores for matrix[i,j]
    #  and matrix[j,i]
    n = matrix.shape[0]
    for node in range(n):
        node_children = children(matrix, node)
        for child in node_children:
            if matrix[child, node] == 0:
                matrix[child, node] = matrix[node, child]

    return matrix
