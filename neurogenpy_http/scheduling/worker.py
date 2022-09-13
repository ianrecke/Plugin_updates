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
              estimation: str, data_type: str, own: bool):
    import statistics
    import pandas as pd
    from neurogenpy import BayesianNetwork, GEXF, JSON

    hostname = log_rec(parcellation_id, roi, genes, algorithm, estimation,
                       data_type)

    try:
        if own:
            df = pd.read_csv('df.csv')

        else:
            import siibra

            parcellation = siibra.parcellations[parcellation_id]
            region = parcellation.decode_region(roi)
            if region is None:
                logger.warning(
                    f'Region definition {roi} could not be matched in atlas.')

            samples = {
                gene_name: [statistics.mean(f.expression_levels) for f in
                            siibra.get_features(region, 'gene',
                                                gene=gene_name)] for
                gene_name in genes}

            df = pd.DataFrame(samples)

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
def perform_inference(json_bn: str, evidence: dict, own: bool):
    from neurogenpy import JSON

    hostname = log_rec(evidence)

    try:
        if own:
            result = {
                "marginals": {
                    "GLUL": {
                        "No-change": 1
                    },
                    "GEMIN2": {
                        "No-change": 1
                    },
                    "C15orf27": {
                        "Inhibition": 0.006384753970989123,
                        "No-change": 0.9936152460290109
                    },
                    "ARHGAP28": {
                        "Inhibition": 0.045028450074193774,
                        "No-change": 0.9549715499258062
                    },
                    "NINL": {
                        "No-change": 1
                    },
                    "RBM15": {
                        "No-change": 1
                    },
                    "A_24_P127063": {
                        "No-change": 1
                    },
                    "LOC728327": {
                        "No-change": 1
                    },
                    "A_24_P738859": {
                        "Inhibition": 0.014061688594069525,
                        "No-change": 0.9859383114059305
                    },
                    "RBP5": {
                        "Inhibition": 0.042054828277381655,
                        "No-change": 0.9579451717226183
                    },
                    "ACOT6": {
                        "Activation": 0.14751969521949823,
                        "Inhibition": 0.242090128990776,
                        "No-change": 0.6103901757897259
                    },
                    "A_24_P945069": {
                        "Activation": 0.09594943760822791,
                        "Inhibition": 0.21190870275860485,
                        "No-change": 0.6921418596331673
                    },
                    "LARP4": {
                        "No-change": 1
                    },
                    "TXLNG": {
                        "No-change": 1
                    },
                    "A_24_P485271": {
                        "Inhibition": 0.053908481383679686,
                        "No-change": 0.9460915186163203
                    },
                    "A_23_P32821": {
                        "Activation": 0.07397942942685658,
                        "Inhibition": 0.13833913900636122,
                        "No-change": 0.7876814315667823
                    },
                    "LOC728804": {
                        "Activation": 0.003332071655053021,
                        "Inhibition": 0.06284449348314998,
                        "No-change": 0.933823434861797
                    },
                    "A_24_P713312": {
                        "No-change": 1
                    },
                    "A_24_P791814": {
                        "Activation": 0.11984642813170517,
                        "Inhibition": 0.23742873800194908,
                        "No-change": 0.6427248338663457
                    },
                    "SNHG8": {
                        "No-change": 1
                    },
                    "A_24_P118946": {
                        "No-change": 1
                    },
                    "RDH12": {
                        "Inhibition": 0.006444365801265232,
                        "No-change": 0.9935556341987347
                    },
                    "IKBKAP": {
                        "No-change": 1
                    },
                    "PRSS50": {
                        "No-change": 1
                    },
                    "ATP2A3": {
                        "Activation": 0.054518148233292825,
                        "Inhibition": 0.12635468058929783,
                        "No-change": 0.8191271711774094
                    },
                    "PAMR1": {
                        "No-change": 1
                    },
                    "A_24_P878561": {
                        "Activation": 0.03933329240757433,
                        "Inhibition": 0.13295606531028953,
                        "No-change": 0.8277106422821361
                    },
                    "NGEF": {
                        "No-change": 1
                    },
                    "SPATA7": {
                        "No-change": 1
                    },
                    "GPS1": {
                        "No-change": 1
                    },
                    "PNMT": {
                        "Inhibition": 0.040572362840001,
                        "No-change": 0.959427637159999
                    },
                    "ERICH5": {
                        "Inhibition": 0.012880569161721993,
                        "No-change": 0.987119430838278
                    },
                    "PRKAA2": {
                        "No-change": 1
                    },
                    "C5orf58": {
                        "Inhibition": 0.12129572847516279,
                        "No-change": 0.8787042715248372
                    },
                    "A_24_P942374": {
                        "Activation": 0.0032314914406388253,
                        "Inhibition": 0.03532293309788976,
                        "No-change": 0.9614455754614714
                    },
                    "A_32_P12494": {
                        "Activation": 0.025851931525110602,
                        "Inhibition": 0.19877584957874392,
                        "No-change": 0.7753722188961455
                    },
                    "A_24_P911051": {
                        "Activation": 0.06473698303607524,
                        "Inhibition": 0.16645962732919253,
                        "No-change": 0.7688033896347322
                    },
                    "A_24_P654368": {
                        "Activation": 0.0032289812922058716,
                        "Inhibition": 0.08184692265933967,
                        "No-change": 0.9149240960484545
                    },
                    "C19orf67": {
                        "Activation": 0.05115391116312463,
                        "Inhibition": 0.15399549145467056,
                        "No-change": 0.7948505973822048
                    },
                    "A_24_P232763": {
                        "No-change": 1
                    },
                    "NADK2": {
                        "No-change": 1
                    },
                    "BTBD16": {
                        "Activation": 0.009724628238452505,
                        "Inhibition": 0.10249116776587937,
                        "No-change": 0.8877842039956682
                    },
                    "ASPG": {
                        "Activation": 0.012860105005171957,
                        "Inhibition": 0.009651567837653108,
                        "No-change": 0.9774883271571749
                    },
                    "FLJ20021": {
                        "No-change": 1
                    },
                    "A_24_P791862": {
                        "Activation": 0.0691105359046114,
                        "Inhibition": 0.1664596273291926,
                        "No-change": 0.7644298367661961
                    },
                    "TNFRSF10C": {
                        "Activation": 0.04522768266248311,
                        "Inhibition": 0.0685105562739565,
                        "No-change": 0.8862617610635604
                    },
                    "UBXN4": {
                        "No-change": 1
                    },
                    "A_23_P64051": {
                        "Activation": 0.01598200109852398,
                        "Inhibition": 0.01923626897804251,
                        "No-change": 0.9647817299234336
                    },
                    "LOC728836": {
                        "Activation": 0.051339540907947706,
                        "Inhibition": 0.07800940911525953,
                        "No-change": 0.8706510499767928
                    },
                    "A_24_P487877": {
                        "Activation": 0.0415009008114293,
                        "Inhibition": 0.07563970814842148,
                        "No-change": 0.8828593910401492
                    },
                    "A_24_P938284": {
                        "Activation": 0.03870091779970208,
                        "Inhibition": 0.19589775387324124,
                        "No-change": 0.7654013283270567
                    },
                    "KCNC3": {
                        "No-change": 1
                    },
                    "CALM3": {
                        "No-change": 1
                    },
                    "STGC3": {
                        "Activation": 0.0064311895071535825,
                        "Inhibition": 0.03212596651833864,
                        "No-change": 0.9614428439745077
                    },
                    "A_24_P934744": {
                        "Activation": 0.00334448160535117,
                        "Inhibition": 0.03837910775321191,
                        "No-change": 0.958276410641437
                    },
                    "A_32_P48054": {
                        "Activation": 0.04808492988139429,
                        "Inhibition": 0.17885468568698382,
                        "No-change": 0.773060384431622
                    },
                    "FCHSD2": {
                        "No-change": 1
                    },
                    "A_24_P222054": {
                        "No-change": 1
                    },
                    "DSCAML1": {
                        "No-change": 1
                    },
                    "A_32_P170564": {
                        "Activation": 0.046057975193747554,
                        "Inhibition": 0.12972606735443207,
                        "No-change": 0.8242159574518204
                    },
                    "PIK3CA": {
                        "No-change": 1
                    },
                    "FBXW11": {
                        "No-change": 1
                    },
                    "A_24_P213073": {
                        "No-change": 1
                    },
                    "A_32_P208978": {
                        "No-change": 1
                    },
                    "A_24_P136911": {
                        "Activation": 0.08031446034116793,
                        "Inhibition": 0.15460151503616051,
                        "No-change": 0.7650840246226716
                    },
                    "DPYSL5": {
                        "No-change": 1
                    },
                    "MAPK10": {
                        "No-change": 1
                    },
                    "A_24_P936419": {
                        "No-change": 1
                    },
                    "ITGB8": {
                        "No-change": 1
                    },
                    "KIF18A": {
                        "Activation": 0.05303979725472368,
                        "Inhibition": 0.08633412228650389,
                        "No-change": 0.8606260804587724
                    },
                    "MED12L": {
                        "No-change": 1
                    },
                    "MGARP": {
                        "Inhibition": 0.0032149584038410504,
                        "No-change": 0.996785041596159
                    },
                    "A_32_P40327": {
                        "No-change": 1
                    },
                    "MEF2C": {
                        "No-change": 1
                    },
                    "LOC100132014": {
                        "Activation": 0.06705239337824143,
                        "Inhibition": 0.15664459763838645,
                        "No-change": 0.7763030089833721
                    },
                    "A_23_P21804": {
                        "No-change": 1
                    },
                    "A_24_P934971": {
                        "Activation": 0.0865556849912654,
                        "Inhibition": 0.1408109873587082,
                        "No-change": 0.7726333276500265
                    },
                    "OPTC": {
                        "Activation": 0.03429536996715795,
                        "Inhibition": 0.10214924665886052,
                        "No-change": 0.8635553833739815
                    },
                    "A_24_P221724": {
                        "Activation": 0.0707414508015468,
                        "Inhibition": 0.15780592988412415,
                        "No-change": 0.7714526193143291
                    },
                    "MLH1": {
                        "No-change": 1
                    },
                    "HSPA1B": {
                        "No-change": 1
                    },
                    "A_23_P26367": {
                        "Activation": 0.006590273977991002,
                        "Inhibition": 0.03824547026614418,
                        "No-change": 0.9551642557558648
                    },
                    "A_32_P5148": {
                        "Activation": 0.030100334448160532,
                        "Inhibition": 0.17725752508361203,
                        "No-change": 0.7926421404682275
                    },
                    "A_24_P608931": {
                        "Activation": 0.01938361894749645,
                        "Inhibition": 0.12522366069717286,
                        "No-change": 0.8553927203553308
                    },
                    "LMAN1L": {
                        "Activation": 0.0096264796647296,
                        "No-change": 0.9903735203352704
                    },
                    "HNRNPCL1": {
                        "Inhibition": 0.006556525863522313,
                        "No-change": 0.9934434741364777
                    },
                    "ADCY10P1": {
                        "Activation": 0.08643003220541098,
                        "Inhibition": 0.12442260040604396,
                        "No-change": 0.7891473673885452
                    },
                    "OR11H4": {
                        "Activation": 0.028569056669046165,
                        "Inhibition": 0.06913698250321217,
                        "No-change": 0.9022939608277417
                    },
                    "ACOT12": {
                        "Activation": 0.07703224659198757,
                        "Inhibition": 0.09074159569390074,
                        "No-change": 0.8322261577141117
                    },
                    "AP4S1": {
                        "Activation": 0.0032220298785390292,
                        "No-change": 0.996777970121461
                    },
                    "SCRN1": {
                        "No-change": 1
                    },
                    "CNTFR": {
                        "Activation": 0.022448649272532338,
                        "Inhibition": 0.07147996717757948,
                        "No-change": 0.9060713835498883
                    },
                    "A_32_P219635": {
                        "No-change": 1
                    },
                    "A_32_P140153": {
                        "No-change": 1
                    },
                    "RPL27": {
                        "No-change": 1
                    },
                    "A_24_P340886": {
                        "No-change": 1
                    },
                    "CDY1": {
                        "Activation": 0.03793492699369431,
                        "Inhibition": 0.14661115282233292,
                        "No-change": 0.8154539201839727
                    },
                    "COPA": {
                        "No-change": 1
                    },
                    "A_24_P299137": {
                        "No-change": 1
                    },
                    "A_24_P938281": {
                        "No-change": 1
                    },
                    "SKAP2": {
                        "No-change": 1
                    },
                    "CLEC6A": {
                        "Activation": 0.03189593281342261,
                        "Inhibition": 0.11537667613580409,
                        "No-change": 0.8527273910507732
                    },
                    "DLX2": {
                        "No-change": 1
                    },
                    "A_24_P75856": {
                        "Activation": 0.070384335576586,
                        "Inhibition": 0.16688667451678735,
                        "No-change": 0.7627289899066267
                    },
                    "SLC12A8": {
                        "No-change": 1
                    },
                    "MSANTD4": {
                        "No-change": 1
                    },
                    "SEPT4": {
                        "No-change": 1
                    },
                    "KANSL1L": {
                        "No-change": 1
                    },
                    "ZNF700": {
                        "No-change": 1
                    },
                    "CREM": {
                        "No-change": 1
                    },
                    "TBC1D24": {
                        "No-change": 1
                    },
                    "A_24_P489399": {
                        "No-change": 1
                    },
                    "WBSCR17": {
                        "No-change": 1
                    },
                    "A_24_P592487": {
                        "No-change": 1
                    },
                    "ARL5A": {
                        "No-change": 1
                    },
                    "PZP": {
                        "Activation": 0.08421430153187863,
                        "Inhibition": 0.10577668872821429,
                        "No-change": 0.810009009739907
                    },
                    "A_24_P485105": {
                        "Activation": 0.00334448160535117,
                        "No-change": 0.9966555183946488
                    },
                    "C8orf22": {
                        "Activation": 0.055708226946484786,
                        "Inhibition": 0.12922702298351352,
                        "No-change": 0.8150647500700017
                    },
                    "A_24_P392622": {
                        "Activation": 0.17882178973284454,
                        "Inhibition": 0.2399237084484183,
                        "No-change": 0.5812545018187372
                    },
                    "ZNF484": {
                        "No-change": 1
                    },
                    "ATP5G1": {
                        "No-change": 1
                    },
                    "LINC01420": {
                        "No-change": 1
                    },
                    "PSG2": {
                        "Activation": 0.03536016882158305,
                        "Inhibition": 0.09654738588767445,
                        "No-change": 0.8680924452907426
                    },
                    "GOLPH3": {
                        "No-change": 1
                    },
                    "GGT7": {
                        "No-change": 1
                    },
                    "CCDC64B": {
                        "No-change": 1
                    },
                    "MMP11": {
                        "Activation": 0.009630800344370255,
                        "Inhibition": 0.03222029878539029,
                        "No-change": 0.9581489008702394
                    },
                    "LINC00165": {
                        "Activation": 0.12409539888530319,
                        "Inhibition": 0.18660496273484048,
                        "No-change": 0.6892996383798563
                    },
                    "MAPRE1": {
                        "No-change": 1
                    },
                    "ACACA": {
                        "No-change": 1
                    },
                    "MXRA8": {
                        "No-change": 1
                    },
                    "SMIM19": {
                        "No-change": 1
                    },
                    "BSCL2": {
                        "No-change": 1
                    },
                    "RALB": {
                        "No-change": 1
                    },
                    "ADGRG6": {
                        "Activation": 0.009632582290828225,
                        "Inhibition": 0.009667228091107356,
                        "No-change": 0.9807001896180645
                    },
                    "TM7SF3": {
                        "No-change": 1
                    },
                    "EEF2K": {
                        "Inhibition": 0.008984746897279948,
                        "No-change": 0.9910152531027201
                    },
                    "PGBD5": {
                        "No-change": 1
                    },
                    "A_24_P178643": {
                        "No-change": 1
                    },
                    "RAB33B": {
                        "No-change": 1
                    },
                    "AADAT": {
                        "No-change": 1
                    },
                    "CUST_2664_PI416261804": {
                        "No-change": 1
                    },
                    "ANKRD36C": {
                        "No-change": 1
                    },
                    "SNRNP40": {
                        "No-change": 1
                    },
                    "NOSIP": {
                        "No-change": 1
                    },
                    "A_23_P210451": {
                        "Activation": 0.12216263428213685,
                        "Inhibition": 0.24124531897951806,
                        "No-change": 0.636592046738345
                    },
                    "ARHGAP1": {
                        "No-change": 1
                    },
                    "A_32_P193792": {
                        "No-change": 1
                    },
                    "C7orf13": {
                        "No-change": 1
                    },
                    "TRIP13": {
                        "Activation": 0.03857744540066263,
                        "Inhibition": 0.12243531154329831,
                        "No-change": 0.8389872430560391
                    },
                    "PRKCZ": {
                        "No-change": 1
                    },
                    "SPDYE2B": {
                        "No-change": 1
                    },
                    "A_23_P96262": {
                        "Activation": 0.09975622081999765,
                        "Inhibition": 0.08747288083852768,
                        "No-change": 0.8127708983414746
                    },
                    "A_24_P612020": {
                        "No-change": 1
                    },
                    "ARMCX6": {
                        "No-change": 1
                    },
                    "WTIP": {
                        "No-change": 1
                    },
                    "ZNF730": {
                        "No-change": 1
                    },
                    "OR2B3": {
                        "Activation": 0.050808602532748486,
                        "Inhibition": 0.11559075119637953,
                        "No-change": 0.833600646270872
                    },
                    "A_32_P51005": {
                        "No-change": 1
                    },
                    "NEIL3": {
                        "Activation": 0.04455050977125104,
                        "Inhibition": 0.11227898365960118,
                        "No-change": 0.8431705065691478
                    },
                    "A_32_P97968": {
                        "Activation": 0.07926183095340478,
                        "Inhibition": 0.15821404915154824,
                        "No-change": 0.762524119895047
                    },
                    "GAGE10": {
                        "Activation": 0.06990265263303805,
                        "Inhibition": 0.11461320742149854,
                        "No-change": 0.8154841399454634
                    },
                    "C6orf62": {
                        "No-change": 1
                    },
                    "L3MBTL1": {
                        "No-change": 1
                    },
                    "TRAF3IP3": {
                        "Activation": 0.00334448160535117,
                        "Inhibition": 0.03871823881379544,
                        "No-change": 0.9579372795808534
                    },
                    "HYMAI": {
                        "Activation": 0.08349009399692998,
                        "Inhibition": 0.14169791882259258,
                        "No-change": 0.7748119871804774
                    },
                    "TOLLIP": {
                        "No-change": 1
                    },
                    "TP53I13": {
                        "Inhibition": 0.015645231903743635,
                        "No-change": 0.9843547680962563
                    },
                    "FAM21C": {
                        "Inhibition": 0.003189593281342261,
                        "No-change": 0.9968104067186577
                    },
                    "A_23_P93109": {
                        "Activation": 0.0032513052678698998,
                        "Inhibition": 0.07692400341707756,
                        "No-change": 0.9198246913150525
                    },
                    "SLC6A7": {
                        "Activation": 0.003212579872577504,
                        "Inhibition": 0.04813802830014946,
                        "No-change": 0.948649391827273
                    },
                    "A_24_P706236": {
                        "Activation": 0.043806649330482186,
                        "Inhibition": 0.13578246900791285,
                        "No-change": 0.8204108816616049
                    },
                    "A_32_P114372": {
                        "No-change": 1
                    },
                    "A_24_P161733": {
                        "No-change": 1
                    },
                    "DGCR6": {
                        "No-change": 1
                    },
                    "NDUFB5": {
                        "No-change": 1
                    },
                    "KIAA0922": {
                        "Activation": 0.012869248446409054,
                        "Inhibition": 0.061005179882595253,
                        "No-change": 0.9261255716709956
                    },
                    "GNMT": {
                        "Activation": 0.02278508415152818,
                        "Inhibition": 0.09297211249771926,
                        "No-change": 0.8842428033507526
                    },
                    "DENND2A": {
                        "No-change": 1
                    },
                    "COX15": {
                        "No-change": 1
                    },
                    "CYP3A4": {
                        "Activation": 0.00964487521152315,
                        "Inhibition": 0.061118876805582895,
                        "No-change": 0.929236247982894
                    },
                    "ERG": {
                        "No-change": 1
                    },
                    "A_24_P384379": {
                        "Activation": 0.12891981316420786,
                        "Inhibition": 0.18087282882259065,
                        "No-change": 0.6902073580132014
                    },
                    "CYP3A43": {
                        "Inhibition": 0.01602800540550183,
                        "No-change": 0.9839719945944981
                    },
                    "PPA2": {
                        "No-change": 1
                    },
                    "MOSPD3": {
                        "No-change": 1
                    },
                    "KLHL34": {
                        "Activation": 0.0032087213067467316,
                        "No-change": 0.9967912786932533
                    },
                    "IGFBP5": {
                        "No-change": 1
                    },
                    "SUSD6": {
                        "No-change": 1
                    },
                    "NAA20": {
                        "No-change": 1
                    },
                    "TGIF2": {
                        "Activation": 0.051390901141806014,
                        "Inhibition": 0.09996656854644702,
                        "No-change": 0.848642530311747
                    },
                    "RHOU": {
                        "No-change": 1
                    },
                    "CUST_2438_PI416261804": {
                        "Inhibition": 0.009637888514020754,
                        "No-change": 0.9903621114859792
                    },
                    "LNPEP": {
                        "No-change": 1
                    },
                    "TEX101": {
                        "Inhibition": 0.03529896876284235,
                        "No-change": 0.9647010312371576
                    },
                    "A_32_P463538": {
                        "Inhibition": 0.018203559454339123,
                        "No-change": 0.981796440545661
                    },
                    "OR7E91P": {
                        "Inhibition": 0.003216317862442413,
                        "No-change": 0.9967836821375575
                    },
                    "TEAD2": {
                        "Activation": 0.04413812434004661,
                        "Inhibition": 0.12550863606639798,
                        "No-change": 0.8303532395935554
                    }
                }
            }
        else:
            bn = JSON().convert(json_bn)

            bn.clear_evidence()
            bn.set_evidence(evidence)
            new_marginals = bn.condition()
            result = {'marginals': new_marginals}

        log_success(hostname, result)

        return result

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
