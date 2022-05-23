import pandas as pd

from neurogenpy import BayesianNetwork

# Using io subpackage

# Using BayesianNetwork
# %%
# Instance a particular
# :class:`~neurogenpy.structure.learn_structure.LearnStructure` or
# :class:`~neurogenpy.parameters.learn_parameters.LearnParameters` subclass:
# Loading an already saved Bayesian network:
#
# BIF file (pgmpy): it loads the graph structure and the parameters of the
# network.
bn = BayesianNetwork().load('bn.bif')

# %%
# GEXF (graph stored with .gexf extension), CSV (adjacency matrix stored with
# '.csv') or parquet (adjacency matrix stored with '.gzip' extension) file, it
# only loads the graph structure of the network. The parameters can be learnt
# according to this graph and the initial data.
bn = BayesianNetwork().load('bn.gexf')
df = pd.read_csv('file.csv')
bn = bn.fit(df, estimation='mle', skip_structure=True)
