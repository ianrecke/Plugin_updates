import pandas as pd
import numpy as np
import networkx
import os
import tempfile

from .IOBN import IOBN


class AdjMatrix(IOBN):
    def __init__(self, file_extension):
        super(AdjMatrix, self).__init__()
        self.extension = file_extension

    def import_file(self, file_path):
        if self.extension == "csv":
            pd_adj_matrix = pd.read_csv(file_path, na_filter=False,
                                        dtype=np.float64, low_memory=False)
        elif self.extension == "gzip":
            pd_adj_matrix = pd.read_parquet(file_path,
                                            engine="fastparquet").astype(
                np.float64)

        nodes_names = pd_adj_matrix.columns.values[1:]
        adj_matrix = pd_adj_matrix.iloc[:, 1:].values
        bn_graph = networkx.from_numpy_matrix(adj_matrix,
                                              create_using=networkx.DiGraph)

        mapping = {}
        for i, node in enumerate(bn_graph.nodes()):
            mapping[node] = nodes_names[i]

        bn_graph = networkx.relabel_nodes(bn_graph, mapping)

        return bn_graph

    def export_file(self, file_path, bn):
        file_path = file_path + "." + self.extension
        file_path = os.path.join(tempfile.gettempdir(), file_path)

        adj_matrix = networkx.to_numpy_matrix(bn.graph, bn.graph.nodes())
        pd_adj_matrix = pd.DataFrame(adj_matrix, columns=bn.graph.nodes())
        pd_adj_matrix.to_csv(file_path)

        with open(file_path) as file_exported:
            file_exported_memory = file_exported.read()

        return file_exported_memory
