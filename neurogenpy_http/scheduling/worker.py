import logging
import os
from typing import List

import numpy as np
import pandas as pd

CHANNEL = os.getenv('NEUROGENPY_CELERY_CHANNEL', 'neurogenpy_http')

logger = logging.getLogger(__name__)

try:
    from celery import Celery
except ImportError as e:
    logger.critical(f'Importing celery error')
    raise e

default_config = 'conf.celeryconfig'
app = Celery(CHANNEL)
app.config_from_object(default_config)


@app.task
def learn_grn(parcellation_id: str, roi: str, genes: List[str], algorithm: str,
              estimation: str):
    import siibra
    from neurogenpy import BayesianNetwork
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
        samples = {gene_name: [np.mean(f.expression_levels) for f in
                               siibra.get_features(region, 'gene',
                                                   gene=gene_name)] for
                   gene_name in genes}

        df = pd.DataFrame(samples)

        bn = BayesianNetwork().fit(df, algorithm=algorithm,
                                   estimation=estimation)

        result = 'bn.gexf'
        bn.save(file_path='bn.gexf')

        logger.info(f'{hostname}:task:success')
        logger.debug(f'{hostname}:task:success_result {result}')
        return {'result': result}

    except Exception as exc:
        logger.critical(f'{hostname}:task:failed {str(exc)}')
        raise exc
