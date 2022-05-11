"""
FGES-Merge structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:


from operator import itemgetter
from tempfile import TemporaryDirectory

import numpy as np
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from neurogenpy.utils.data_structures import matrix2nx
from .fges import FGESStructure, FGESBase
from ..io.adjacency_matrix import save_tmp
from ..utils.graph import get_hubs, force_directions, union, \
    remove_unrelated, remove_children_edges, intersect, remove_local_unrelated
from ..utils.statistics import hypothesis_test_related_genes


class FGESMerge(FGESBase):
    """
    FGES-Merge structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    penalty : int, default=45
        Penalty hyperparameter of the FGES algorithm.

    References
    ----------
    .. [1] N. Bernaola, M. Michiels, P. Larrañaga, C. Bielza, Learning massive
       interpretable gene regulatory networks of the human brain by merging
       Bayesian Networks, bioRxiv
       `<https://doi.org/10.1101/2020.02.05.935007>`_.
       `<https://www.biorxiv.org/content/early/2020/02/05/2020.02.05.935007>`_.
    """

    def __init__(self, df, data_type, *, penalty=45, **_):
        super().__init__(df, data_type, penalty=penalty)

        self.tmp_dir = TemporaryDirectory()
        self.graph_files = {}

    def run(self, env='neurogenpy'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'neurogenpy'}, default='neurogenpy'
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
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):
        """
        """

        self._setup()
        # Assign n/size (rounded to an integer) nodes to each process
        # if rank == 0:
        #     chunks_nodes_bics = []
        #     chunks_nodes = [
        #         self.nodes[int(i * self.num_nodes / size):int(
        #             (i + 1) * self.num_nodes / size)] for i in range(size)]
        #
        #     for chunk in chunks_nodes:
        #         this_chunk = {}
        #         for node in chunk:
        #             this_chunk[node] = self.bics[node, :]
        #         chunks_nodes_bics.append(this_chunk)
        # else:
        #     chunks_nodes_bics = None

        # TODO: establish parallelization strategy: parallelize local
        #  network learning or arrows forward calculation, but it does not
        #  make sense to do both.

        with logging_redirect_tqdm():
            for node in tqdm(self.nodes):
                neighbors, ndata, nbics = self._neighborhood_candidates(node)

                # Construct local network with node and neighborhood
                # candidates:
                fges_structure = FGESStructure(ndata, nbics, neighbors,
                                               self.penalty, self.n_jobs)
                local_graph = fges_structure.run()

                self.graph_files[node] = save_tmp(local_graph, neighbors,
                                                  self.tmp_dir.name)

        config = {
            'combine_method': 'union',
            'backward_intersect': True,
            'backward_intersect_global_bad': False,
            'remove_unrelated': True,
            'force_dirs_before': True,
            'force_dirs_after': True,
            'remove_edges_before': True,
            'remove_edges_after': True,
            'threshold': 91,
            'hubs_method': 'degree',
            'n_parents': 3,
            'counting': False,
        }

        self._run_combination(config)

        nx_graph = matrix2nx(self.graph, self.nodes_ids)
        self.tmp_dir.cleanup()

        return nx_graph

    def _neighborhood_candidates(self, node):
        """
        Returns the neighbor candidates for a node. They can not be more than
        100.

        Parameters
        ----------
        node :
            The node whose neighbors have to be retrieved.

        Returns
        -------
            A 3-tuple formed by: a list with the selected nodes, the data
            associated with them and the bics for them.
        """
        max_neighbors = 100
        positive_bics = [(j, self.bics[j]) for j in self.nodes if
                         self.bics[j] > 0]
        positive_bics.sort(key=itemgetter(1), reverse=True)

        positive_bics_values = [pb[1] for pb in positive_bics]

        max_candidates = min(max_neighbors, len(positive_bics_values))
        num_candidates = hypothesis_test_related_genes(
            max_candidates, positive_bics_values)

        neighbors = [candidate[0] for candidate in
                     positive_bics[:num_candidates]]
        neighbors.append(node)

        neighbors_data = self.data[:, neighbors]

        return neighbors, neighbors_data, self.bics[
            np.ix_(neighbors, neighbors)]

    def _run_combination(self, config):
        """
        Parallelizes subgraphs combination when possible.

        Parameters
        ----------
        config : dict
            Combination configuration.
        """

        if self.n_jobs > 1:
            avg_size = self.num_nodes / self.n_jobs
            nodes_chunks = [
                list(range(int(i * avg_size), int((i + 1) * avg_size)))
                for i in range(self.n_jobs)]

            nodes_subgraphs = [[self.graph_files[i] for i in chunk] for chunk
                               in nodes_chunks]

            subgraphs = []
            with MPIPoolExecutor() as executor:
                for result in executor.map(self._combine_subgraphs,
                                           nodes_subgraphs, config):
                    subgraphs.append(result)

            self._combine_subgraphs(subgraphs, config, final=True)
        else:
            self._combine_subgraphs(list(self.graph_files.values()), config,
                                    final=True)

    def _combine_subgraphs(self, subgraphs, config, final=False):
        """
        Combines a set of subgraphs with the combination configuration
        provided. If `final` is True, the obtained combination is set as the
        final learnt graph. Otherwise, the graph is stored and will be used to
        obtain the final one.

        Parameters
        ----------
        subgraphs :
            Subgraphs to combine.

        config : dict
            Combination configuration.

        final : bool, default=False
            Whether the graph to compute is the final learnt graph or not.

        Returns
        -------
        str
            In case the graph is not final, it returns the name of the file
            where the obtained graph is stored.
        """

        total_nodes = set()
        data = {}

        for graph_file in subgraphs:
            with np.load(graph_file) as npz:
                local_nodes = npz['nodes']
                data[graph_file] = local_nodes, npz['graph']
                total_nodes += set(local_nodes)
        total_nodes = list(total_nodes)
        global_graph = np.zeros((len(total_nodes), len(total_nodes)))

        for graph_file in subgraphs:
            # TODO: Check if keeping all the subgraphs in memory is OK.
            local_nodes = data[graph_file][0]
            local_graph = data[graph_file][1]

            if config['counting']:
                local_graph = (local_graph > 0).astype(np.int64)

            if config['force_dirs_before']:
                hubs = get_hubs(local_graph, config['threshold'],
                                method=config['hubs_method'])
                local_graph = force_directions(local_graph, hubs)
                if config['remove_edges_before']:
                    local_graph = remove_children_edges(local_graph, hubs,
                                                        config['n_parents'])

            if config['combine_method'] == 'union':
                global_graph = union(global_graph, local_graph, local_nodes)
            else:
                global_graph = intersect(global_graph, local_graph,
                                         local_nodes)

        if config['backward_intersect']:
            for graph_file in subgraphs:
                local_nodes = data[graph_file][0]
                local_graph = data[graph_file][1]

                global_graph = remove_local_unrelated(
                    global_graph, local_graph, local_nodes,
                    config['backward_intersect_global_bad'])

        if config['counting']:
            global_graph = global_graph / len(subgraphs)

        if config['remove_unrelated']:
            global_graph = remove_unrelated(global_graph)

        if config['force_dirs_after']:
            hubs = get_hubs(global_graph, config['threshold'],
                            method=config['hubs_method'])
            global_graph = force_directions(global_graph, hubs)
            if config['remove_edges_children_after']:
                global_graph = remove_children_edges(global_graph, hubs,
                                                     config['n_parents'])

        if final:
            self.graph = global_graph
        else:
            return save_tmp(global_graph, total_nodes, self.tmp_dir.name,
                            delete=True)
