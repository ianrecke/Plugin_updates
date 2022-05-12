import pandas as pd

from neurogenpy import BayesianNetwork

df = pd.read_csv('../files_examples/datasets/asia10k.csv', dtype='category')

df.pop('id')

bn = BayesianNetwork().fit(df=df, estimation='mle', algorithm='MMHC',
                           data_type='discrete')

bn.save(layout='circular')
