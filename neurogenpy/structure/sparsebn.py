"""
sparsebn structure learning module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import networkx as nx
import numpy as np
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure, pd2r


class SparseBn(LearnStructure):
    """
    sparsebn structure learning class.
    """

    def run(self, env='sparsebn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'sparsebn'}, default='sparsebn'
            Environment used to run the algorithm.

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        """

        if env == 'neurogenpy':
            return self._run_neurogenpy()
        elif env == 'sparsebn':
            return self._run_sparsebn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_sparsebn(self):
        """

        """
        if self.data_type == 'hybrid':
            raise ValueError(
                'This algorithm does not support hybrid Bayesian networks')

        dataframe = pd2r(self.data)

        sparsebn = importr('sparsebn')
        sparsebn_utils = importr('sparsebnUtils')
        data_r = sparsebn_utils.sparsebnData(dataframe, type=self.data_type)
        dags_r = sparsebn.estimate_dag(data_r)
        solution_idx_r = sparsebn_utils.select_parameter(dags_r, data_r)
        output_raw_r = sparsebn_utils.select(dags_r, index=solution_idx_r)
        output_raw_r = dict(output_raw_r.items())

        edges_r = np.array(output_raw_r['edges'])
        nodes = output_raw_r['edges'].names
        edges_python = []
        for i, parents in enumerate(edges_r):
            parents_node_i = list(parents)
            for parent in parents_node_i:
                edge = (nodes[parent], nodes[i])
                edges_python.append(edge)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges_python)

        return graph

    def _run_neurogenpy(self):
        return None
