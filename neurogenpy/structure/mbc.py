"""
Multidimensional Bayesian network classifier structure learning module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class MBC(LearnStructure):
    """
    Multidimensional Bayesian network classifier class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    Raises
    ------
    ValueError
        If `features_classes` is empty.
    """

    def __init__(self, df, data_type, *, features_classes=None, **_):

        super().__init__(df, data_type)
        self.features_classes = features_classes
        if len(self.features_classes) == 0:
            raise ValueError(
                'To run this classifier, you must supply at least one class '
                'feature in the previous section.')

    def run(self, env='bnlearn'):
        """
        Learns the structure of the Bayesian network.

        Parameters
        ----------
        env : {'bnlearn', 'neurogenpy'}, default='bnlearn'
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
        elif env == 'bnlearn':
            return self._run_mbc_bnlearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_mbc_bnlearn(self):
        """

        """
        features = list(
            set(self.data.columns.values) - set(self.features_classes))

        # Black list of arcs from features to classes
        blacklist = pd.DataFrame(columns=['from', 'to'])
        blacklist['from'] = features * len(self.features_classes)
        blacklist['to'] = np.repeat(
            self.features_classes, [
                len(features)], axis=0)

        # Learn MBC structure
        return self._run_bnlearn(importr('bnlearn').hc, blacklist=blacklist)

    def _run_neurogenpy(self):

        return None
