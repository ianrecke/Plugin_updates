import numpy as np
from rpy2.robjects.packages import importr

from .learn_structure import LearnStructure


class NB(LearnStructure):
    """
    Naive Bayes structure learning class.

    Parameters
    ----------
    df : pandas.DataFrame
        Data set with the learning sample from which to infer the network.

    data_type : {'continuous', 'discrete' or 'hybrid'}
        Type of the data introduced.

    features_classes :

    Raises
    ------
    ValueError
        If `features_classes` is empty.
    """

    def __init__(self, df, data_type, *, features_classes, **_):

        super().__init__(df, data_type)
        self.features_classes = features_classes
        if len(self.features_classes) == 0:
            raise Exception(
                'To run this classifier, you must supply one class feature in '
                'the previous section.')

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
            return self._run_nb_bnlearn()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_nb_bnlearn(self):
        """

        """
        explanatory = list(self.data.columns.values)

        try:
            explanatory.remove(self.features_classes[0])
        except ValueError:
            pass
        explanatory = np.array(explanatory)

        return self._run_bnlearn(importr('bnlearn').naive_bayes,
                                 training=self.features_classes[0],
                                 explanatory=explanatory)

    def _run_neurogenpy(self):

        return None
