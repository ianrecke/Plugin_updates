"""
FGES base classes module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from abc import ABCMeta
from itertools import permutations

import numba
import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from .learn_structure import LearnStructure
from ..utils.fges_arrows import create_arrow, get_nodes, get_dest, get_vals, \
    get_bic
from ..utils.graph import add_edge, adjacencies, remove_undirected, \
    parents, children, undirected_neighbors, add_undirected, exists_sd_path, \
    is_undirected, remove_edge, has_edge


@numba.jit(nopython=True, fastmath=True)
def _bic(y, X=None, penalty=45):
    """
    Calculates the BIC score between the nodes in X and the node in y. It uses
    numba in order to speed up this computation.

    Parameters
    ----------
    X :
        Data for the nodes.
    y :
        Data for the node.
    penalty :
        Penalty hyperparameter.

    Returns
    -------
    float
        BIC score.
    """
    n = y.shape[0]

    if X is None:
        return -n * np.log(np.sum(np.square(y - np.mean(y)) / n))

    k = X.shape[1]
    X = np.ascontiguousarray(X).reshape(n, -1)
    y = np.ascontiguousarray(y).reshape(n, -1)

    A = np.ones((n, k + 1), dtype=np.float64)
    A[:, :k] = X

    return - penalty * k * np.log(n) - n * np.log(
        np.sum(np.square(y - np.dot(np.linalg.lstsq(A, y)[0].T, A.T).T)) / n)


@numba.jit(nopython=True)
def _parts_of(nodes, max_size=None):
    """

    Parameters
    ----------
    nodes

    max_size : int, optional

    Returns
    -------

    """

    max_size = len(nodes) if max_size is None else min(max_size, len(nodes))

    subsets = []
    nodes = np.array(list(nodes), dtype=np.int64)

    for sz in range(max_size):
        subsets.extend(_permutations(nodes, sz + 1))

    return subsets


@numba.jit(nopython=True)
def _permutations(A, k):
    # From https://github.com/numba/numba/issues/3599
    r = [[i for i in range(0)]]
    for i in range(k):
        r = [[a] + b for a in A for b in r if (a in b) is False]

    # Remove repetitions:
    sets = []
    for combo_r in r:
        set_r = set(combo_r)
        append = True
        for saved_set in sets:
            if set_r == saved_set:
                append = False
                break
        if append:
            sets.append(set_r)

    return sets


class FGESStructure:
    """FGES structure class. It is used in `FGES` and `FGES-Merge` classes."""

    def __init__(self, data, bics, nodes, penalty, n_jobs):
        self.data = np.array(data, dtype=np.float64)
        self.bics = bics
        self.num_nodes = len(nodes)
        self.nodes = nodes
        self.graph = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)
        self.arrows = []
        self.penalty = penalty
        self.n_jobs = n_jobs

    def run(self):
        """
        Learns the structure of the Bayesian network.
        """

        self._fes()
        self.arrows = []
        self._reevaluate_backward(self.nodes)
        self._bes()
        self._orient_graph()
        return self._get_final_graph()

    # TODO: Check temperature and delta usage.
    def _fes(self, stochastic=True, temperature=None, delta=None):
        """Forward equivalence search."""

        self.arrows = [create_arrow(i, j, set(), set(), self.bics[i, j]) for
                       i in range(self.num_nodes) for j in
                       range(self.num_nodes) if self.bics[i, j] > 0]

        if stochastic and temperature is None:
            temperature = self.num_nodes ** 2
        while self.arrows:
            arrows_bics = np.array([get_bic(ar) for ar in self.arrows],
                                   dtype=np.float64)
            max_index = np.argmax(arrows_bics)
            if stochastic:
                best_bic = np.max(arrows_bics)
                rand_bic = np.random.choice(arrows_bics)
                index_rand = np.where(arrows_bics == rand_bic)[0][0]
                if temperature > 0 and np.random.random(1) <= np.exp(
                        (rand_bic - best_bic) / temperature):
                    best_edge = self.arrows.pop(index_rand)

                else:
                    best_edge = self.arrows.pop(max_index)
            else:
                best_edge = self.arrows.pop(max_index)

            x, y, NaYX, T, b = get_vals(best_edge)

            NaYX_T = NaYX.union(T)

            if self._check_arrow(best_edge) and self._check_clique(
                    NaYX_T) and not exists_sd_path(self.graph, y, x,
                                                   set(T).union(NaYX)):
                self.bics[x, y] = 0
                self.bics[y, x] = 0

                # Add edge
                self.graph = add_edge(self.graph, x, y)
                if stochastic:
                    temperature = temperature - self.num_nodes if \
                        delta is None else temperature - delta

                self._orient_T_into_y(T, y)
                nodes = self._apply_meek_rules_locally({x, y})
                self._reevaluate_forward(nodes)

    def _reevaluate_backward(self, nodes):
        """"""

        for node in nodes:
            node_parents = parents(self.graph, node)
            for parent in node_parents:
                self.arrows = [arrow for arrow in self.arrows if
                               get_nodes(arrow) not in [(parent, node),
                                                        (node, parent)]]
                self.arrows.extend(
                    self._calculate_arrows_backward(parent, node))

            neighbors = undirected_neighbors(self.graph, node)
            for neighbor in neighbors:
                if neighbor < node:
                    continue
                self.arrows = [arrow for arrow in self.arrows if
                               get_nodes(arrow) not in [(neighbor, node),
                                                        (node, neighbor)]]
                self.arrows.extend(
                    self._calculate_arrows_backward(neighbor, node))
                self.arrows.extend(
                    self._calculate_arrows_backward(node, neighbor))

    def _bes(self):
        """Backward equivalence search."""

        while self.arrows:
            arrows_bics = np.array([get_bic(ar) for ar in self.arrows],
                                   dtype=np.float64)
            max_index = np.argmax(arrows_bics)
            best_edge = self.arrows.pop(max_index)

            x, y, NaYX, S, b = get_vals(best_edge)

            NaYX_S = NaYX - S
            if x in children(self.graph, y):
                continue

            adj_y = adjacencies(self.graph, y)
            adj_x = adjacencies(self.graph, x)
            neighbors_y = undirected_neighbors(self.graph, y)
            if self._check_clique(NaYX_S) and x in adj_y and NaYX == (
                    neighbors_y & adj_x):
                if is_undirected(self.graph, x, y):
                    self.graph = remove_undirected(self.graph, x, y)
                elif has_edge(self.graph, x, y):
                    self.graph = remove_edge(self.graph, x, y)

                else:
                    continue

                for node in S:
                    self._orient_T_into_y({x, y}, node)

                nodes = self._apply_meek_rules_locally(
                    {x, y}.union(NaYX_S))
                self._reevaluate_backward(nodes)

    def _orient_graph(self):
        """Orients undirected edges to go from CPDAG to DAG."""

        for node in self.nodes:
            for neighbor in undirected_neighbors(self.graph, node):
                self.graph = remove_undirected(self.graph, node, neighbor)
                if exists_sd_path(self.graph, node, neighbor, set(),
                                  self.graph.shape[0]):
                    self.graph = add_edge(self.graph, node, neighbor)
                else:
                    self.graph = add_edge(self.graph, neighbor, node)

    def _get_final_graph(self):
        """Calculates the final graph structure edges values, i.e., the final
        BIC score for each edge in the graph."""

        graph = np.zeros(self.graph.shape)
        data_len = self.data.shape[0]
        for node in self.nodes:
            node_parents = parents(self.graph, node)
            node_data = self.data[:, node].reshape(data_len, -1)
            X = self.data[:, list(node_parents)].reshape(data_len, -1)
            bic_x = _bic(X, node_data, self.penalty)
            for parent in node_parents:
                S = list(node_parents - {parent})
                X_0 = self.data[:, S].reshape(data_len, -1)
                bic_x0 = _bic(X_0, node_data, self.penalty)
                graph[parent, node] = bic_x - bic_x0

        return graph

    def _check_arrow(self, arrow):
        """
        Checks if an arrow is correct according to the current graph, i.e., for
        an arrow (x, y), NaYX, T, b:

            - x and y are not adjacent.
            - NaYX (neighbors of y adjacent to x) is the same as in the current
                graph.
            - T (neighbors of y not adjacent to x) is contained in the T
                neighbors of current graph.

        Parameters
        ----------
        arrow :
            An arrow to be checked.
        """
        x, y, NaYX, T, b = get_vals(arrow)

        neighbors_y = undirected_neighbors(self.graph, y)
        adj_x = adjacencies(self.graph, x)

        return y not in adj_x and NaYX == (neighbors_y & adj_x) and T <= (
                neighbors_y - adj_x)

    def _orient_T_into_y(self, T, y):
        """Establishes edges from each node of `T` to `y`."""

        for node in T:
            if is_undirected(self.graph, node, y):
                self.graph = remove_undirected(self.graph, node, y)
            elif has_edge(self.graph, node, y):
                self.graph = remove_edge(self.graph, node, y)

            self.graph = add_edge(self.graph, node, y)

    def _apply_meek_rules_locally(self, nodes):
        """Transform to CPDAG and correct possible cycles between the
        V-structures of the CPDAG."""

        cpdag_changes, non_v_edges = self._cpdag(nodes)
        nodes = nodes.union(cpdag_changes)
        changes = nodes.copy()

        while changes:
            changes = self._apply_meek_rules(changes)
            nodes = nodes.union(changes)

        new_non_v_edges = self._non_v_structures(nodes)

        if new_non_v_edges == non_v_edges:
            return nodes
        else:
            return self._apply_meek_rules_locally(nodes)

    def _reevaluate_forward(self, nodes):
        """Recalculate all possible edge additions towards every node in the
        node set."""

        arrows = [arrow for arrow in self.arrows if
                  get_dest(arrow) not in nodes]

        new_arrs = self._calculate_arrows_forward(nodes)
        arrows.extend(new_arrs)

        self.arrows = arrows

    def _calculate_arrows_backward(self, x, y):
        """"""

        arrows = []
        NaYX = undirected_neighbors(self.graph, y) & adjacencies(self.graph, x)
        for subset in _parts_of(NaYX):
            S = NaYX - subset

            if self._check_clique(S):
                S = list(S.union(parents(self.graph, y)) - {x})
                data_y = self.data[:, y]
                X_0 = self.data[:, S]
                X = self.data[: S + [x]]
                bic_x0 = _bic(X_0, data_y, self.penalty)
                bic_x = _bic(X, data_y, self.penalty)
                b = bic_x0 - bic_x

                if b > 0:
                    arrows.append(create_arrow(x, y, NaYX, subset, b))
        return arrows

    def _cpdag(self, nodes):
        """"""

        edges = self._non_v_structures(nodes)
        changes = self._remove_orientation(edges)
        return changes, edges

    def _apply_meek_rules(self, nodes):
        """
        Applies Meek's rules to the current graph and returns the nodes
        modified during the process.

        Parameters
        ----------
        nodes :
        """

        changes = set()

        for node in nodes:
            if len(adjacencies(self.graph, node)) < 2:
                continue
            prev_parents = parents(self.graph, node)
            for parent in prev_parents:
                # Rule 1: Away from collider
                prev_undirected = undirected_neighbors(self.graph, node)
                for neighbor in prev_undirected:
                    if not (neighbor in adjacencies(self.graph, parent)):
                        self.graph = remove_undirected(self.graph, node,
                                                       neighbor)
                        self.graph = add_edge(self.graph, node, neighbor)
                        changes.add(neighbor)
                        changes.add(node)

                # Rule 2: Away from cycle
                prev_children = children(self.graph, node)
                for child in prev_children:
                    if child in undirected_neighbors(self.graph, parent):
                        self.graph = remove_undirected(self.graph, child,
                                                       parent)
                        self.graph = add_edge(self.graph, parent, child)
                        changes.add(child)
                        changes.add(parent)

            # Rule 3: Double triangle
            prev_undirected = undirected_neighbors(self.graph, node)
            kite_changes = set()
            if len(prev_undirected) < 3:
                continue
            kite_permutations = permutations(prev_undirected, 3)
            kite_permutations = (perm for perm in kite_permutations if
                                 perm[0] > perm[2])
            for node_b, node_c, node_d in kite_permutations:
                if node_b not in adjacencies(self.graph, node_d):
                    if node_c in (
                            children(self.graph, node_b) & children(self.graph,
                                                                    node_d)):
                        try:
                            self.graph = remove_undirected(self.graph, node,
                                                           node_c)
                        except ValueError:
                            if node_c in kite_changes:
                                continue
                            else:
                                raise KeyError

                        self.graph = add_edge(self.graph, node, node_c)
                        kite_changes.add(node)
                        kite_changes.add(node_c)
            changes.update(kite_changes)
        return changes

    def _non_v_structures(self, nodes):
        """
        Finds the set of all edges that are not involved in unshielded
        colliders in the current graph.

        Parameters
        ----------
        nodes :
            Set of nodes to check.

        Returns
        -------
            A set of edges not involved in unshielded colliders.
        """
        edges = set()

        for node in nodes:
            nodes = nodes.union(parents(self.graph, node))

        for node in nodes:
            ch = children(self.graph, node)
            for child in ch:
                v_struct = False
                parents_child = parents(self.graph, child) - {node}
                for pc in parents_child:
                    if pc not in adjacencies(self.graph, node):
                        v_struct = True
                        break
                    else:
                        continue
                if not v_struct:
                    edges.add((node, child))

        return edges

    def _calculate_arrows_forward(self, nodes):

        if self.n_jobs > 1:
            arrows = self._caf_mpi(nodes)
        else:
            nodes_pbs = {node: pbs for node in nodes if (
                pbs := [j for j in self.nodes if self.bics[j, node] > 0])}

            arrows = self._caf(nodes_pbs)

        return [create_arrow(*arr) for arr in arrows]

    def _remove_orientation(self, edges):
        """
        Removes the orientation of some edges converting them in undirected
        edges.

        Parameters
        ----------
        edges :
            A set of edges to transform.

        Returns
        -------
            The set of nodes involved in modifications.
        """
        changes = set()
        for x, y in edges:
            self.graph = add_undirected(self.graph, x, y)
            changes.add(y)
            changes.add(x)
        return changes

    def _caf_mpi(self, nodes):
        """
        """

        chunks_pbs = []
        chunks = np.array_split(nodes, self.n_jobs)
        for chunk in chunks:
            pbs = {node: pbs for node in chunk if
                   (pbs := [j for j in self.nodes if self.bics[j, node] > 0])}
            chunks_pbs.append(pbs)

        arrows = []
        with MPIPoolExecutor() as executor:
            for result in executor.map(self._caf, chunks_pbs):
                arrows.extend(result)

        return arrows

    def _caf(self, nodes_pbs):
        """

        Parameters
        ----------
        nodes_pbs :

        Returns
        -------
        """
        arrows = []
        for node, positive_scores in nodes_pbs.items():
            arrows.extend(self._caf_node(node, positive_scores))
        return arrows

    @numba.jit(nopython=True)
    def _check_clique(self, nodes):
        """
        Checks if a set of nodes form a clique in the current graph, i.e., the
        subgraph formed by them is complete. It uses numba in order to
        speed up calculations.

        Parameters
        ----------
        nodes :
            A set of nodes in the graph.
        """
        # TODO: Check if flatten and reshape (redundant) are needed.
        nodes = set(nodes.reshape(nodes.shape[0], -1).flatten())
        for node in nodes:
            if (adjacencies(self.graph, node).intersection(nodes)).union(
                    {node}) < nodes:
                return False
        return True

    @numba.jit(nopython=True, parallel=True)
    def _caf_node(self, node, positive_bics):
        """
        Calculates the arrows forward for a node given the positive BIC score
        differences for it.

        Parameters
        ----------
        node :

        positive_bics :
        """

        # TODO: Ask about max_size use.
        max_size = 3

        arrows = []
        n = len(positive_bics)
        for i in numba.prange(n):
            x = positive_bics[i]

            T = undirected_neighbors(self.graph, node) - adjacencies(
                self.graph, x)

            NaYX = undirected_neighbors(self.graph, node) & adjacencies(
                self.graph, x)

            subsets = _parts_of(T, max_size)
            for j in numba.prange(len(subsets)):
                subset = subsets[j]

                S = NaYX.union(subset)
                S_ = np.array(list(S), dtype=np.int64)

                if self._check_clique(S_):
                    S = S.union(parents(self.graph, node))
                    node_data = self.data[:, node]

                    array_S = np.array(list(S), dtype=np.int64)
                    array_S_X = np.array(list(S) + [x], dtype=np.int64)

                    X_0 = self.data[:, array_S]
                    X = self.data[:, array_S_X]

                    bic_X = _bic(X, node_data, self.penalty)
                    bic_X_0 = _bic(X_0, node_data, self.penalty)
                    b = bic_X - bic_X_0

                    if b > 0:
                        arrows.append((x, node, NaYX, subset, b))

        return arrows


class FGESBase(LearnStructure, metaclass=ABCMeta):
    """FGES base class. It is superclass of `FGES` and `FGES-Merge` classes."""

    def __init__(self, df, data_type, *, penalty=45, **_):
        super().__init__(df, data_type)
        self.penalty = penalty
        self.num_nodes = self.data.shape[1]
        self.nodes = list(range(self.num_nodes))
        self.nodes_ids = list(df.columns.values)
        self.bics = np.empty((self.num_nodes, self.num_nodes),
                             dtype=np.float64)
        self.n_jobs = MPI.UNIVERSE_SIZE
        self.graph = None

    def _init_bics_mpi(self):
        """
        Initializes BIC scores using mpi4py.
        """

        # Get the nodes_chunk. The optimal number of operations for each
        # process is n*(n-1)/(2*n_jobs) and we try to get as close as possible:
        # we assign nodes to a process until this amount is reached.
        ops_per_chunk = np.floor(
            (self.num_nodes ** 2 - self.num_nodes) / (2 * self.n_jobs))
        nodes_chunks = []
        running_count = 0
        limiter = 0
        for i in range(self.num_nodes):
            running_count += self.num_nodes - 1 - i
            if running_count >= ops_per_chunk:
                nodes_chunks.append(list(range(limiter, i + 1)))
                limiter = i + 1
                running_count = 0
        if limiter != self.num_nodes:
            nodes_chunks.append(list(range(limiter, self.num_nodes)))

        # TODO: set max_workers for MPIPoolExecutor.
        with MPIPoolExecutor() as executor:
            # TODO: check if the creation of chunks and use of map is needed or
            #   we can just use submit.
            for result in executor.map(self._init_bics, nodes_chunks):
                result = np.append(
                    np.zeros(len(result), self.num_nodes - result.shape[1]),
                    result, axis=1)
                self.bics = np.append(self.bics, result, axis=0)

        self.bics = self.bics + np.triu(self.bics, k=1).T

    @numba.jit(nopython=True, parallel=True)
    def _init_bics(self, nodes):
        """
        Calculates the local BIC of adding each node from nodes_chunk to its
        following nodes.

        Parameters
        ----------
        nodes :
            Nodes used for the computation.

        Returns
        -------
        numpy.array
            BIC scores matrix.
        """

        bics = np.zeros([len(nodes), (self.num_nodes - nodes[0])],
                        dtype=np.float64)
        for i in numba.prange(len(nodes)):
            node_i = nodes[i]
            X = self.data[:, node_i]
            j = 0
            for node_j in numba.prange(node_i + 1, self.num_nodes):
                y = self.data[:, node_j]
                bic_x = _bic(X, y, self.penalty)
                bic_y = _bic(y)
                bics[i, j] = bic_x - bic_y
                j += 1

        return bics

    def _setup(self):
        """

        """

        # Numba setup
        # https://numba.pydata.org/numba-doc/dev/reference/envvars.html
        # https://numba.pydata.org/numba-doc/dev/user/threading-layer.html#numba-threading-layer
        numba.config.THREADING_LAYER = 'omp'
        numba.config.NUMBA_WARNINGS = 1
        # TODO: set the number of threads in a way we avoid
        #  oversubscription.
        # env['NUMBA_NUM_THREADS'] = str(numba.config.NUMBA_DEFAULT_NUM_THREADS)
        # numba.config.reload_config()

        # TODO: get bics initialization if available (previously run).
        if self.n_jobs > 1:
            self._init_bics_mpi()
        else:
            self._init_bics(self.nodes)
