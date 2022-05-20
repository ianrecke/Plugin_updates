"""
Tree augmented naive Bayes structure learning module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class Tan(LearnStructure):
    """
    Tree augmented naive Bayes structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    features_classes : list

    Raises
    ------
    ValueError
        If `features_classes` is empty.
    """

    def __init__(self, df, data_type, *, features_classes=None):

        super().__init__(df, data_type)
        self.features_classes = features_classes
        if len(self.features_classes) == 0:
            raise ValueError(
                'To run this classifier, you must supply one class feature.')

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : str, default='bnlearn'
            Environment used to run the algorithm. Currently supported:

                - 'bnlearn': :cite:`naive_bayes`

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
        elif env == 'bnlearn':
            explanatory = list(self.data.columns.values)

            try:
                explanatory.remove(self.features_classes[0])
            except ValueError as _:
                pass
            explanatory = np.array(explanatory)

            return self._run_bnlearn(importr('bnlearn').tree_bayes,
                                     training=self.features_classes[0],
                                     explanatory=explanatory)
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):

        return None
