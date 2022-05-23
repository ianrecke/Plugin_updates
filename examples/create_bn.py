import pandas as pd
from networkx import DiGraph

from neurogenpy import BayesianNetwork, GaussianNode

# %%
# The use of the package is focused on the
# :class:`~neurogenpy.models.bayesian_network.BayesianNetwork` class.
#
# If you already have a graph structure and the network parameters (or joint
# probability distribution) in the right formats, it is posible to use the
# constructor for building the network. See
# :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.fit` and
# :func:`~neurogenpy.models.bayesian_network.BayesianNetwork.load` methods for
# other ways of creating Bayesian networks.
graph = DiGraph()
graph.add_nodes_from([1, 2])
graph.add_edges_from([(1, 2)])
ps = {1: GaussianNode(0, 1, [], []), 2: GaussianNode(0, 1, [1], [0.8])}
bn = BayesianNetwork(graph=graph, parameters=ps)

# %%
# Learning the structure and parameters of a Bayesian network from the data
# in a CSV file.
#
# Set the structure and parameter learning methods with arguments:
df = pd.read_csv('../files_examples/datasets/asia10k.csv', dtype='category')
df.pop('id')
bn = BayesianNetwork().fit(df, estimation='mle', algorithm='PC')

# %%
# Additional parameters for the structure learning or parameters estimation
# algorithm can be provided too:
bn = BayesianNetwork()
bn = bn.fit(df, algorithm='FGESMerge', penalty=45)