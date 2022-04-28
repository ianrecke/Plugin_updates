"""
Data structures utilities module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import networkx as nx
import numpy as np
from igraph import Graph
from pgmpy.models import BayesianModel
from rpy2.robjects import pandas2ri


def get_data_type(df):
    """
    Retrieves the data type of the full data set which types are provided.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set to be checked.

    Returns
    -------
    ({'continuous', 'discrete', 'hybrid'}, list)
        The type of the data set and the set of continuous variables if the
        type is 'hybrid'.
    """

    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    if all(is_number(df.dtypes)):
        return 'continuous', None
    elif (not any(is_number(df.dstypes))) and (
            any(df.dtypes == 'object') or any(df.dtypes == 'bool')):
        return 'discrete', None
    else:
        aux = [x for x in df.columns if np.issubdtype(df[x].dtype, np.number)]
        return 'hybrid', aux


def pd2r(data):
    """
    Converts a pandas DataFrame into an R dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Data set to be transformed.

    Returns
    -------
        R dataframe with the same information as the input dataframe.
    """
    # TODO: check pandas DataFrame to R dataframe conversion with activate
    pandas2ri.activate()
    dataframe = data

    return dataframe


def matrix2nx(graph, nodes):
    """
    Converts an adjacency matrix into a networkx graph.

    Parameters
    ----------
    graph : numpy.array
        Adjacency matrix of the graph.

    nodes : list
        IDs of the nodes in the adjacency matrix.

    Returns
    -------
    networkx.DiGraph
        A graph representing the network determined by the adjacency matrix.
    """

    mapping = {i: node for i, node in enumerate(nodes)}
    nx_graph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph())
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    return nx_graph


def nx2pgmpy(graph, parameters):
    """
    Converts a networkx graph and some distribution parameters into a pgmpy
    Bayesian model.

    Parameters
    ----------
    graph : networkx.DiGraph

    parameters : dict

    Returns
    -------
    pgmpy.BayesianModel
    """

    pgmpy_model = BayesianModel()
    pgmpy_model.add_nodes_from(graph.nodes())
    pgmpy_model.add_edges_from(graph.edges())
    if parameters:
        pgmpy_model.add_cpds(*parameters)

    return pgmpy_model


def pgmpy2nx(bn):
    """
    Converts a pgmpy Bayesian model into an adjacency matrix and distribution
    parameters.

    Parameters
    ----------
    bn : pgmpy.BayesianModel

    Returns
    -------
    (networkx.DiGraph, dict)
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(bn.nodes())
    graph.add_edges_from(bn.edges())
    parameters = bn.get_cpds()

    return graph, parameters


def nx2igraph(graph):
    """
    Converts a NetworkX DiGraph into an iGraph graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The NetworkX DiGraph to convert.

    Returns
    -------
    igraph.Graph
        The equivalent iGraph graph.
    """

    result = Graph(directed=True)
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    result.add_vertices(nodes)
    result.add_edges(edges)

    return result


def bnlearn2nx(nodes, r_output):
    """
    Converts R's bnlearn output into a NetworkX graph.

    Parameters
    ----------
    nodes: list
        Nodes that form the network.

    r_output:
        R's bnlearn output.

    Returns
    -------
    networkx.DiGraph
        A graph representing the network.
    """

    arcs_r = r_output.rx2('arcs')
    edges_from = np.array(arcs_r.rx(True, 1))
    edges_to = np.array(arcs_r.rx(True, 2))
    edges = zip(edges_from, edges_to)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
