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

BN = None


@app.task
def learn_grn(parcellation_id: str, roi: str, genes: List[str], algorithm: str,
              estimation: str):
    import siibra
    import statistics
    import pandas as pd
    from neurogenpy import BayesianNetwork, GEXF

    global BN

    hostname = log_rec(parcellation_id, roi, genes, algorithm, estimation)

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
        # df = pd.read_csv('df.csv')

        BN = BayesianNetwork().fit(df, algorithm=algorithm,
                                   estimation=estimation)

        gexf = GEXF(BN).generate(layout_name='circular', communities=True)
        marginals = BN.all_cpds()
        # marginals = {node: BN.marginal([node]) for node in BN.nodes()}

        log_success(hostname, gexf, marginals)
        return {'gexf': gexf, 'marginals': marginals}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def get_related(node, method):
    global BN

    hostname = log_rec(node, method)

    try:
        result = []
        if method == 'mb':
            result = BN.markov_blanket(node)
        elif method == 'reachable':
            result = list(BN.reachable_nodes([node]))

        log_success(hostname, result)
        return {'result': result}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def get_layout(layout):
    from neurogenpy.io.layout import DotLayout, IgraphLayout

    hostname = log_rec(layout)

    try:
        lo = IgraphLayout(
            BN.graph, layout_name=layout) if layout != "Dot" else DotLayout(
            BN.graph)
        layout_pos = lo.run()

        log_success(hostname, layout_pos)
        return {'result': layout_pos}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def check_dseparation(X, Y, Z):
    hostname = log_rec(X, Y, Z)

    try:

        result = BN.is_dseparated(X, Y, Z)

        log_success(hostname, result)
        return {'result': result}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def perform_inference(evidence, marginals):
    hostname = log_rec(evidence, marginals)

    try:
        BN.set_evidence(evidence)
        non_evidence = [node for node in BN.nodes() if
                        node not in evidence.keys()]
        new_marginals = {node: BN.marginal([node], initial=False) for node in
                         non_evidence}

        for node in evidence.keys():
            new_marginals[node] = {"mu": evidence[node], "sigma": 0}

        log_success(hostname, new_marginals)

        return {'marginals': new_marginals}

    except Exception as exc:
        log_fail(hostname, str(exc))
        raise exc


@app.task
def downloadable_file(file_type, positions):
    hostname = log_rec(file_type)

    try:
        from neurogenpy import JSON, GEXF, AdjacencyMatrix, BIF

        writers = {'json': JSON, 'gexf': GEXF, 'csv': AdjacencyMatrix,
                   'bif': BIF}

        writer = writers[file_type](BN)

        positions = {k: (v['x'], v['y']) for k, v in positions.items()}
        args = {'layout': positions} if file_type == 'gexf' else {}
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
