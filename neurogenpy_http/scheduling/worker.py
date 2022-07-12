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
              estimation: str, data_type: str):
    import siibra
    import statistics
    import pandas as pd
    # import numpy as np
    from neurogenpy import BayesianNetwork, GEXF, JSON

    hostname = log_rec(parcellation_id, roi, genes, algorithm, estimation,
                       data_type)

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
        # rng = np.random.default_rng()
        # df = pd.DataFrame(rng.integers(0, 100, size=(100, 4)),
        #                   columns=list('ABCD'))
        if data_type == 'discrete':
            df = df.apply(lambda col: pd.cut(
                col, bins=[-float('inf'), 2 ** (-0.5) * col.mean(),
                           2 ** 0.5 * col.mean(), float('inf')],
                labels=['Inhibition', 'No-change', 'Activation']))

        class_gene = genes[0] if algorithm in ['nb, tan, mc'] else None

        bn = BayesianNetwork().fit(df=df, data_type=data_type,
                                   estimation=estimation, algorithm=algorithm,
                                   class_variable=class_gene,
                                   class_variables=[class_gene])

        gexf = GEXF(bn).generate(layout_name='circular')
        marginals = bn.all_marginals()

        log_success(hostname, gexf, marginals)
        return {'json_bn': JSON(bn).generate(), 'gexf': gexf,
                'marginals': marginals}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def get_related(json_bn: str, node: str, method: str):
    from neurogenpy import JSON

    hostname = log_rec(node, method)

    try:

        bn = JSON().convert(json_bn)

        result = []
        if method == 'mb':
            result = bn.markov_blanket(node)
        elif method == 'reachable':
            result = list(bn.reachable_nodes([node]))

        log_success(hostname, result)
        return {'result': result}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def get_layout(json_bn: str, layout: str):
    from neurogenpy.io.layout import DotLayout, IgraphLayout
    from neurogenpy import JSON

    hostname = log_rec(layout)

    try:
        bn = JSON().convert(json_bn)

        lo = IgraphLayout(
            bn.graph, layout_name=layout) if layout != "Dot" else DotLayout(
            bn.graph)
        layout_pos = lo.run()

        log_success(hostname, layout_pos)
        return {'result': layout_pos}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def check_dseparation(json_bn: str, X: list, Y: list, Z: list):
    from neurogenpy import JSON

    hostname = log_rec(X, Y, Z)

    try:
        bn = JSON().convert(json_bn)

        result = bn.is_dseparated(X, Y, Z)

        log_success(hostname, result)
        return {'result': result}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def perform_inference(json_bn: str, evidence: dict):
    from neurogenpy import JSON

    hostname = log_rec(evidence)

    try:
        bn = JSON().convert(json_bn)

        bn.clear_evidence()
        bn.set_evidence(evidence)
        new_marginals = bn.condition()

        log_success(hostname, new_marginals)

        return {'marginals': new_marginals}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def downloadable_file(json_bn: str, file_type: str, positions: dict,
                      colors: dict):
    from neurogenpy import JSON, GEXF, AdjacencyMatrix, BIF

    hostname = log_rec(file_type)

    try:
        bn = JSON().convert(json_bn)

        writers = {'json': JSON, 'gexf': GEXF, 'csv': AdjacencyMatrix,
                   'bif': BIF}

        writer = writers[file_type](bn)

        positions = {k: (v['x'], v['y']) for k, v in positions.items()}
        args = {'layout': positions,
                'colors': colors} if file_type == 'gexf' else {}
        result = writer.generate(**args)

        log_success(hostname, result)

        return {'result': result}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


def log_rec(*args):
    import socket
    hostname = socket.gethostname()
    logger.info(f'{hostname}:task:rec')
    logger.debug(
        f'{hostname}:task:rec_param {args}')

    return hostname


def log_success(hostname, *args):
    logger.info(f'{hostname}:task:success')
    logger.debug(f'{hostname}:task:success_result {args}')


def log_fail(hostname, *args):
    logger.critical(f'{hostname}:task:failed {args}')
