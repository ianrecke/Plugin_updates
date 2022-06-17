import logging
import os
from typing import List

CHANNEL = os.getenv('NEUROGENPY_CELERY_CHANNEL', 'neurogenpy_http')

logger = logging.getLogger(__name__)

try:
    from celery import Celery
except ImportError as e:
    logger.critical(f'Importing celery error')
    raise e

default_config = 'neurogenpy_http.conf.celeryconfig'
app = Celery(CHANNEL)
app.config_from_object(default_config)


@app.task
def learn_grn(parcellation_id: str, roi: str, genes: List[str], algorithm: str,
              estimation: str):
    import siibra
    import pandas as pd
    import statistics
    from neurogenpy import BayesianNetwork, JSON, GEXF
    import socket

    hostname = socket.gethostname()
    logger.info(f'{hostname}:task:rec')
    logger.debug(
        f'{hostname}:task:rec_param {parcellation_id} {roi} , '
        f'{",".join(genes)}')

    try:
        parcellation = siibra.parcellations[parcellation_id]
        region = parcellation.decode_region(roi)
        if region is None:
            logger.warning(
                f'Region definition {roi} could not be matched in atlas.')

        # FIXME: Too many requests raise an exception.
        samples = {gene_name: [statistics.mean(f.expression_levels) for f in
                               siibra.get_features(region, 'gene',
                                                   gene=gene_name)] for
                   gene_name in genes}

        df = pd.DataFrame(samples)

        bn = BayesianNetwork().fit(df, algorithm=algorithm,
                                   estimation=estimation)

        graphology_options = {'link': 'edges', 'name': 'key'}
        graphology_keys = ['nodes', 'edges']
        json_graph = JSON(bn).generate(options=graphology_options,
                                       keys=graphology_keys)
        gexf = GEXF(bn).generate()
        # marginals = {node: bn.marginal([node]) for node in bn.nodes()}

        logger.info(f'{hostname}:task:success')
        logger.debug(f'{hostname}:task:success_result {json_graph}')
        logger.debug(f'{hostname}:task:success_result {json_graph}')
        return {'json_graph': json_graph, 'gexf': gexf, 'marginals': []}

    except Exception as exc:
        logger.critical(f'{hostname}:task:failed {str(exc)}')
        raise exc


@app.task
def get_mb(graph_json, node):
    from neurogenpy import BayesianNetwork, JSON
    import socket

    hostname = socket.gethostname()
    logger.info(f'{hostname}:task:rec')
    logger.debug(
        f'{hostname}:task:rec_param {graph_json}, {node}')

    try:
        graphology_options = {'link': 'edges', 'name': 'key'}
        graph = JSON().convert(io_object=graph_json,
                               options=graphology_options)
        bn = BayesianNetwork(graph=graph)

        mb = bn.markov_blanket(node)

        logger.info(f'{hostname}:task:success')
        logger.debug(f'{hostname}:task:success_result {mb}')
        return {'mb': mb}

    except Exception as exc:
        logger.critical(f'{hostname}:task:failed {str(exc)}')
        raise exc
