"""
GENIE3 structure learning module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

from multiprocessing import Pool
from operator import itemgetter

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from .learn_structure import LearnStructure
from ..utils.data_structures import matrix2nx


class GENIE3(LearnStructure):
    """
    GENIE3 structure learning class.

    Notes
    -----
    The algorithm implementation has been taken from
    https://github.com/vahuynh/GENIE3/. A detailed explanation of how it
    works can be seen in [1]_.

    References
    ----------
    .. [1] Huynh-Thu, V. A., Irrthum, A., Wehenkel, L., & Geurts, P. (2010).
        Inferring regulatory networks from expression data using tree-based
        methods. PloS one, 5(9), e12776.
    """

    def run(self, env='genie3'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'genie3'}, default='genie3'
            Environment used to run the algorithm.

        Returns
        -------
        networkx.DiGraph
            Learnt graph structure.

        Raises
        ------
        ValueError
            If the environment is not supported.
        Exception
            If the data is not continuous
        """
        if self.data_type != 'continuous':
            raise Exception(
                'Algorithm only supported for continuous datasets ')

        if env == 'genie3':
            return self.run_genie()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def run_genie(self):
        """


        """

        nodes_names = list(self.data.columns.values)
        data_np = np.array(self.data)

        vim = genie3(data_np, nthreads=1)

        adj_matrix = vim
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = np.triu(adj_matrix)

        graph = matrix2nx(adj_matrix, nodes_names)

        return graph


def compute_feature_importances(estimator):
    if isinstance(estimator, RandomForestRegressor) or \
            isinstance(estimator, ExtraTreesRegressor):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        importances = [e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_]
        importances = np.asarray(importances)
        return sum(importances, axis=0) / len(estimator)


def get_link_list(VIM, gene_names=None, regulators='all', maxcount='all',
                  file_name=None):
    """
    Gets the ranked list of (directed) regulatory links.

    Parameters
    ----------

    VIM: numpy.array
        Array as returned by the function GENIE3(), in which the element (i,j)
        is the score of the edge directed from the i-th gene to the j-th gene.

    gene_names: list of strings, optional
        List of length p, where p is the number of rows/columns in VIM,
        containing the names of the genes. The i-th item of gene_names must
        correspond to the i-th row/column of VIM. When the gene names are not
        provided, the i-th gene is named Gi.

    regulators: list of strings, default='all'
        List containing the names of the candidate regulators. When a list of
        regulators is provided, the names of all the genes must be provided
        (in gene_names), and the returned list contains only edges directed
        from the candidate regulators. When regulators is set to 'all', any
        gene can be a candidate regulator.

    maxcount: 'all' or positive integer, default='all'
        Writes only the first maxcount regulatory links of the ranked list.
        When maxcount is set to 'all', all the regulatory links are written.

    file_name: string, optional
        Writes the ranked list of regulatory links to the file file_name.

    Returns
    -------
        The list of regulatory links, ordered according to the edge score.
        Auto-regulations do not appear in the list. Regulatory links with a
        score equal to zero are randomly permuted. In the ranked list of edges,
        each line has format:

            regulator   target gene     score of edge
    """

    # Check input arguments
    if not isinstance(VIM, np.ndarray):
        raise ValueError('VIM must be a square array')
    elif VIM.shape[0] != VIM.shape[1]:
        raise ValueError('VIM must be a square array')

    ngenes = VIM.shape[0]

    if gene_names is not None:
        if not isinstance(gene_names, (list, tuple)):
            raise ValueError(
                'input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError(
                'input argument gene_names must be a list of length p, where p'
                ' is the number of columns/genes in the expression data')

    if regulators is not 'all':
        if not isinstance(regulators, (list, tuple)):
            raise ValueError(
                'input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError(
                'the gene names must be specified (in input argument '
                'gene_names)')
        else:
            s_intersection = set(gene_names).intersection(set(regulators))
            if not s_intersection:
                raise ValueError(
                    'The genes must contain at least one candidate regulator')

    if maxcount is not 'all' and not isinstance(maxcount, int):
        raise ValueError(
            'input argument maxcount must be "all" or a positive integer')

    if file_name is not None and not isinstance(file_name, str):
        raise ValueError('input argument file_name must be a string')

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = range(ngenes)
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if
                     gene in regulators]

    # Get the non-ranked list of regulatory links
    v_inter = [(i, j, score) for (i, j), score in np.ndenumerate(VIM) if
               i in input_idx and i != j]

    # Rank the list according to the weights of the edges
    v_inter_sort = sorted(v_inter, key=itemgetter(2), reverse=True)
    n_inter = len(v_inter_sort)

    # Random permutation of edges with score equal to 0
    flag = 1
    i = 0
    while flag and i < n_inter:
        (tf_idx, target_idx, score) = v_inter_sort[i]
        if score == 0:
            flag = 0
        else:
            i += 1

    if not flag:
        items_perm = v_inter_sort[i:]
        items_perm = np.random.permutation(items_perm)
        v_inter_sort[i:] = items_perm

    # Write the ranked list of edges
    n_to_write = n_inter
    if isinstance(maxcount, int) and 0 <= maxcount < n_inter:
        n_to_write = maxcount

    if file_name:

        outfile = open(file_name, 'w')

        if gene_names is not None:
            for i in range(n_to_write):
                (tf_idx, target_idx, score) = v_inter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                outfile.write('%s\t%s\t%.6f\n' % (
                    gene_names[tf_idx], gene_names[target_idx], score))
        else:
            for i in range(n_to_write):
                (tf_idx, target_idx, score) = v_inter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                outfile.write(
                    'G%d\tG%d\t%.6f\n' % (tf_idx + 1, target_idx + 1, score))

        outfile.close()

    else:

        if gene_names is not None:
            for i in range(n_to_write):
                (tf_idx, target_idx, score) = v_inter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                print('%s\t%s\t%.6f' % (
                    gene_names[tf_idx], gene_names[target_idx], score))
        else:
            for i in range(n_to_write):
                (tf_idx, target_idx, score) = v_inter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                print('G%d\tG%d\t%.6f' % (tf_idx + 1, target_idx + 1, score))


def genie3(expr_data, gene_names=None, regulators='all', tree_method='RF',
           k='sqrt', ntrees=1000, nthreads=1):
    """
    Computation of tree-based scores for all putative regulatory links.

    Parameters
    ----------
    expr_data: numpy.array
        Array containing gene expression values. Each row corresponds to a
        condition and each column corresponds to a gene.

    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data,
        containing the names of the genes. The i-th item of gene_names must
        correspond to the i-th column of expr_data.

    regulators: list of strings, default='all'
        List containing the names of the candidate regulators. When a list of
        regulators is provided, the names of all the genes must be provided
        (in gene_names). When regulators is set to 'all', any gene can be a
        candidate regulator.

    tree-method: 'RF' or 'ET', default='RF'
        Specifies which tree-based procedure is used: either Random Forest
        ('RF') or Extra-Trees ('ET')

    k: 'sqrt', 'all' or a positive integer, default='sqrt'
        Specifies the number of selected attributes at each node of one tree:
        either the square root of the number of candidate regulators ('sqrt'),
        the total number of candidate regulators ('all'), or any positive
        integer.

    ntrees: positive integer, default=1000
        Specifies the number of trees grown in an ensemble.

    nthreads: positive integer, default=1
        Number of threads used for parallel computing

    tree_method:

    Returns
    -------
        An array in which the element (i,j) is the score of the edge directed
        from the i-th gene to the j-th gene. All diagonal elements are set to
        zero (auto-regulations are not considered). When a list of candidate
        regulators is provided, the scores of all the edges directed from a
        gene that is not a candidate regulator are set to zero.

    """

    # Check input arguments
    if not isinstance(expr_data, np.ndarray):
        raise ValueError(
            'expr_data must be an array in which each row corresponds to a '
            'condition/sample and each column corresponds to a gene')

    ngenes = expr_data.shape[1]

    if gene_names is not None:
        if not isinstance(gene_names, (list, tuple)):
            raise ValueError(
                'input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError(
                'input argument gene_names must be a list of length p, where p'
                ' is the number of columns/genes in the expr_data')

    if regulators is not 'all':
        if not isinstance(regulators, (list, tuple)):
            raise ValueError(
                'input argument regulators must be a list of gene names')

        if gene_names is None:
            raise ValueError(
                'the gene names must be specified (in input argument '
                'gene_names)')
        else:
            s_intersection = set(gene_names).intersection(set(regulators))
            if not s_intersection:
                raise ValueError(
                    'the genes must contain at least one candidate regulator')

    if tree_method is not 'RF' and tree_method is not 'ET':
        raise ValueError(
            'input argument tree_method must be "RF" (Random Forests) or "ET" '
            '(Extra-Trees)')

    if k is not 'sqrt' and k is not 'all' and not isinstance(k, int):
        raise ValueError(
            'input argument K must be "sqrt", "all" or a strictly positive '
            'integer')

    if isinstance(k, int) and k <= 0:
        raise ValueError(
            'input argument K must be "sqrt", "all" or a strictly positive '
            'integer')

    if not isinstance(ntrees, int):
        raise ValueError(
            'input argument ntrees must be a strictly positive integer')
    elif ntrees <= 0:
        raise ValueError(
            'input argument ntrees must be a strictly positive integer')

    if not isinstance(nthreads, int):
        raise ValueError(
            'input argument nthreads must be a strictly positive integer')
    elif nthreads <= 0:
        raise ValueError(
            'input argument nthreads must be a strictly positive integer')

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if
                     gene in regulators]

    # Learn an ensemble of trees for each target gene, and compute scores for
    # candidate regulators
    vim = np.zeros((ngenes, ngenes))

    if nthreads > 1:

        input_data = list()
        for i in range(ngenes):
            input_data.append(
                [expr_data, i, input_idx, tree_method, k, ntrees])

        pool = Pool(nthreads)
        all_output = pool.map(wr_genie3_single, input_data)

        for (i, vi) in all_output:
            vim[i, :] = vi

    else:
        for i in range(ngenes):
            vi = genie3_single(expr_data, i, input_idx, tree_method, k, ntrees)
            vim[i, :] = vi

    vim = np.transpose(vim)

    return vim


def wr_genie3_single(args):
    return ([args[1],
             genie3_single(args[0], args[1], args[2], args[3], args[4],
                           args[5])])


def genie3_single(expr_data, output_idx, input_idx, tree_method, K, ntrees):
    ngenes = expr_data.shape[1]

    # Expression of target gene
    output = expr_data[:, output_idx]

    # Normalize output data
    output = output / np.std(output)

    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    expr_data_input = expr_data[:, input_idx]

    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K, int) and K >= len(input_idx)):
        max_features = 'auto'
    else:
        max_features = K

    if tree_method == 'RF':
        tree_estimator = RandomForestRegressor(n_estimators=ntrees,
                                               max_features=max_features)
    elif tree_method == 'ET':
        tree_estimator = ExtraTreesRegressor(n_estimators=ntrees,
                                             max_features=max_features)

    # Learn ensemble of trees
    tree_estimator.fit(expr_data_input, output)

    # Compute importance scores
    feature_importances = compute_feature_importances(tree_estimator)
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi
