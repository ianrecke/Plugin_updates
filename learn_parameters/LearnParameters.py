from abc import ABCMeta, abstractmethod
from rpy2.robjects import pandas2ri


class LearnParameters(metaclass=ABCMeta):
    """
    Base class for all learn parameters classes.
    """

    def __init__(self, data, data_type, graph):
        """
        LearnParameters constructor.
        @param data: Input data used to learn the parameters from.
        @type data: pandas DataFrame
        @param data_type: Type of the data introduced: continuous, discrete or
            hybrid.
        @param graph: Structure of the Bayesian network.
        """
        self.data = data
        self.data_type = data_type
        self.graph = graph

    @abstractmethod
    def run(self):
        """
        Learns the parameters of the Bayesian network.
        """
        pass

    # TODO: use pd2r function from LearnStructure
    def prepare_input_parameters_bnlearn(self):
        pandas2ri.activate()
        dataframe = self.data
        if self.data_type == "hybrid":
            raise Exception(
                "This algorithm still does not support hybrid bayesian networks")
        dataframe.columns = dataframe.columns.str.replace(".", "")
        dataframe.columns = dataframe.columns.str.replace("-", "")

        return dataframe
