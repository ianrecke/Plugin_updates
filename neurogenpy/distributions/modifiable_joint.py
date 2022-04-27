from tempfile import TemporaryDirectory, NamedTemporaryFile

import numpy as np

from .discrete_joint import DiscreteJointDistribution
from .gaussian_joint import GaussianJointDistribution


class ModifiableJointDistribution:
    """
    Distribution class for the Bayesian network. It keeps the initial and
    current distributions.

    Parameters
    ----------
    initial : JointDistribution, optional
        The initial joint probability distribution. If `params` is set, this
        parameter is ignored.

    save_dist : bool, default='False'
        Whether to load and save the distribution in a file each time it is
        used.

    data_type : {'discrete', 'continuous'}
        Type of the joint probability distribution.

    nodes_order : optional
        Topological order of the variables. It is needed to compute the joint
        probability distribution and, after that, it is used to know the
        index of each variable in the corresponding data structure.

    Raises
    ------
    ValueError
        If `data_type` is not supported.
    """

    def __init__(self, data_type, save_dist=False, initial=None,
                 nodes_order=None):
        self.save_dist = save_dist
        self.tmp_dir = TemporaryDirectory()
        self.initial_path = None
        self.path = None
        self.nodes_order = nodes_order
        self.type = data_type
        self.dist = initial
        self.loaded = initial is not None

        if self.save_dist:
            self.save()
            self.dist.clear()

    def from_params(self, params, nodes_order, data_type, save_dist):
        self.type = data_type
        self.nodes_order = nodes_order
        self.save_dist = save_dist
        if self.type == 'discrete':
            self.dist = DiscreteJointDistribution().from_parameters(
                params,
                self.nodes_order)
        elif self.type == 'continuous':
            self.dist = GaussianJointDistribution().from_parameters(
                params,
                self.nodes_order)
        else:
            raise ValueError(f'{data_type} data type is not supported.')

        self.loaded = True

        if self.save_dist:
            self.save()
            self.dist.clear()

    def save(self, initial=False, path=None):
        """
        Saves the joint probability distribution in a .npz file.

        Parameters
        ----------
        initial : bool, default=False
            Whether the distribution to save has to be considered the initial
            joint distribution.

        path : str, optional
            The path where to save the distribution.
        """

        if path is None:
            outfile = NamedTemporaryFile(delete=False, dir=self.tmp_dir)
            path = outfile.name
            if initial:
                self.initial_path = path
            else:
                self.path = path

        np.savez(path, nodes_order=self.nodes_order, **self.dist.__dict__)

    def load(self, initial=False):
        """
        Loads a distribution

        Parameters
        ----------
        initial : bool, default=False
            Whether the distribution to load is the initial distribution.
        """
        if initial:
            config = np.load(self.initial_path)
            self.initial_path = None
        else:
            config = np.load(self.path)
        self.nodes_order = config.pop('nodes_order')
        for k, v in config.items():
            setattr(self.dist, k, v)
        self.path = None

    def restart(self):
        """
        Establishes the initial distribution as the current distribution.
        """

        self.load(initial=True)

        if self.save_dist:
            self.save()
            self.dist.clear()

    def condition(self, evidence):
        """
        Conditions the current joint probability distribution on some evidence.

        Parameters
        ----------
        evidence : dict
            The evidence to use for conditioning the distribution.
        """

        if self.initial_path is not None:
            self.load(initial=True)
        else:
            self.save(initial=True)

        self.dist.condition(evidence, self.nodes_order)
        for elem in evidence.keys():
            self.nodes_order.remove(elem)

        if self.save_dist:
            self.save()
            self.dist.clear()

    def marginal(self, nodes, initial=False):
        """
        Computes the marginal probability distribution for a set of nodes.

        Parameters
        ----------
        nodes : list
            The nodes for which to calculate the marginal probability
            distribution.

        initial : bool, default=False
            Whether to calculate the marginal distribution from the initial
            distribution or not.

        Returns
        -------

        """

        if initial and self.initial_path is not None:
            self.load(initial=initial)
        elif self.save_dist:
            self.load()

        marginal = self.dist.marginal(self.nodes_order, nodes)

        if self.save_dist:
            self.save()
            self.dist.clear()

        return marginal

    def relabel_nodes(self, mapping):
        self.nodes_order = [mapping[node] if node in mapping.keys() else node
                            for node in self.nodes_order]

    def is_set(self):
        return self.loaded
