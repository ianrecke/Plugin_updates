"""
Adjacency matrix input/output module. It mainly uses `networkx` functionality.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from tempfile import NamedTemporaryFile

import networkx
import numpy as np
import pandas as pd

from .bnio import BNIO
from ..util.data_structures import matrix2nx


def save_tmp(graph, nodes, directory, delete=False):
    """
    Saves the numpy array that represents the adjacency matrix of a Bayesian
    network graph structure in a temporary .npz file.

    Parameters
    ----------
    graph : numpy.array
        The graph represented by its adjacency matrix.

    nodes : list
        IDs of the nodes present in the graph.

    directory : str
        The directory used to store the file.

    delete : bool, default=False
        Whether to delete the file after closing it or not.

    Returns
    -------
    str
        The name of the file where the graph is stored.
    """

    outfile = NamedTemporaryFile(delete=delete, dir=directory)
    np.savez(outfile, nodes=np.array(nodes), graph=graph)
    return outfile.name


class AdjacencyMatrix(BNIO):
    """
    Adjacency matrix input/output class.
    """

    def convert(self, io_object, nodes=None):
        """
        Creates the graph structure object from the adjacency matrix received.

        Parameters
        ----------
        io_object : numpy.array

        nodes : list, optional
            Nodes order present in the adjacency matrix.

        Returns
        -------
        networkx.DiGraph
            The graph structure of the loaded Bayesian network.
        """
        if not nodes:
            nodes = list(range(io_object.shape[0]))
        return matrix2nx(io_object, nodes)

    # TODO: Add parquet case
    def generate(self, representation='csv'):
        """
        Generates the adjacency matrix that represents the network. It also
        retrieves the nodes order used to build the matrix.

        Parameters
        ----------
        representation : str, default='csv'

        Returns
        -------
        (numpy.array, list)
            The adjacency matrix for the graph structure and the nodes order
            used to represent it.
        """

        pd_adj_matrix = self._get_df()
        if representation == 'csv':
            return pd_adj_matrix.to_csv(index=False)

    def read_file(self, file_path):
        """
        Returns the graph structure of a Bayesian network whose adjacency
        matrix is stored in a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored. It can be a CSV or
            parquet file.

        Returns
        -------
        networkx.DiGraph
            The graph structure loaded.
        """

        if file_path.endswith('.csv'):
            pd_adj_matrix = pd.read_csv(file_path, na_filter=False,
                                        dtype=np.float64, low_memory=False)
        elif file_path.endswith('.gzip'):
            pd_adj_matrix = pd.read_parquet(file_path,
                                            engine='fastparquet').astype(
                np.float64)
        else:
            raise ValueError('File extension not supported.')

        nodes = pd_adj_matrix.columns.values[1:]
        adj_matrix = pd_adj_matrix.iloc[:, 1:].values

        return matrix2nx(adj_matrix, nodes)

    def write_file(self, file_path):
        """
        Writes a Bayesian network structure adjacency matrix in a file.

        Parameters
        ----------
        file_path: str
            Path of the file to store the Bayesian network in. It can be a CSV
            or parquet file.
        """

        pd_adj_matrix = self._get_df()
        if file_path.endswith('.csv'):
            pd_adj_matrix.to_csv(file_path)
        elif file_path.endswith('.gzip'):
            pd_adj_matrix.to_parquet(file_path)

    def _get_df(self):
        """Returns the DataFrame with the adjacency matrix."""

        nodes = list(self.bn.graph.nodes())
        adj_matrix = networkx.to_numpy_matrix(self.bn.graph, nodes)
        return pd.DataFrame(adj_matrix, columns=nodes)
