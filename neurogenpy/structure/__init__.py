from rpy2.robjects.packages import importr, isinstalled

from .cl import CL
from .fast_iamb import FastIamb
from .fges import FGES
from .fges_merge import FGESMerge
from .genie3 import GENIE3
from .graphical_lasso import GraphicalLasso
from .grow_shrink import GrowShrink
from .hc_tabu import HcTabu
from .hill_climbing import HillClimbing
from .hiton_pc import HitonPC
from .iamb import Iamb
from .inter_iamb import InterIamb
from .learn_structure import LearnStructure
from .lr import Lr
from .mbc import MBC
from .mi_continuous import MiContinuous
from .mmhc import MMHC
from .mmpc import MMPC
from .naive_bayes import NB
from .pc import PC
from .pearson import Pearson
from .sparsebn import SparseBn
from .tan import Tan

__all__ = [
    'CL',
    'FastIamb',
    'FGES',
    'FGESMerge',
    'GENIE3',
    'GraphicalLasso',
    'GrowShrink',
    'HillClimbing',
    'HcTabu',
    'HitonPC',
    'Iamb',
    'InterIamb',
    'Lr',
    'MBC',
    'MiContinuous',
    'MMHC',
    'MMPC',
    'NB',
    'PC',
    'Pearson',
    'SparseBn',
    'Tan',
    'LearnStructure'
]

utils = importr('utils')
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

if not isinstalled('bnlearn'):
    utils.install_packages('bnlearn')
