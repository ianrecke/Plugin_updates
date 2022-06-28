"""
Bayesian network input/output base module. It provides a base class for
input/output formats that are completed using ´networkx` and `pgmpy`
functionality.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from abc import ABCMeta, abstractmethod


class BNIO(metaclass=ABCMeta):
    """
    Base class for all input/output Bayesian network classes.

    Parameters
    ----------
    bn : BayesianNetwork, optional
        The BayesianNetwork needed in some cases.
    """

    def __init__(self, bn=None):
        self.bn = bn

    @abstractmethod
    def read_file(self, file_path):
        """
        Reads a Bayesian network from a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the network is stored.

        Returns
        -------
        networkx.DiGraph or (networkx.DiGraph, dict)
            The graph structure of the loaded Bayesian network and the
            parameters in case the format provides them.
        """

    @abstractmethod
    def write_file(self, file_path):
        """
        Writes a Bayesian network in a file.

        Parameters
        ----------
        file_path :
            Path of the file to store the Bayesian network in.
        """

    @abstractmethod
    def generate(self):
        """
        Generates the object that represents the network.

        Returns
        -------
            The object that represents the network.
        """

    @abstractmethod
    def convert(self, io_object):
        """
        Creates the attributes of a Bayesian network from the input/output
        object received.

        Parameters
        ----------
        io_object
            input/output object.

        Returns
        -------
        networkx.DiGraph or (networkx.DiGraph, dict)
            The graph structure loaded and the parameters in case the format
            provides them.
        """

    @abstractmethod
    def generate(self, bn):
        """
        Generates the object that represents the network.

        Parameters
        ----------
        bn : BayesianNetwork

        Returns
        -------
            The object that represents the network.
        """
