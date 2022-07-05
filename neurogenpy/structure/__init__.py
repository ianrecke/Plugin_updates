# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

import logging
from subprocess import check_call, CalledProcessError

from .fges import FGES
from .fges_merge import FGESMerge
from .genie3 import GENIE3
from .graphical_lasso import GraphicalLasso
from .lr import Lr
from .mbc import MBC
from .mi_continuous import MiContinuous
from .pearson import Pearson

logger = logging.getLogger(__name__)

__all__ = [
    'FGES',
    'FGESMerge',
    'GENIE3',
    'GraphicalLasso',
    'Lr',
    'MBC',
    'MiContinuous',
    'Pearson',
]

try:
    # TODO: Check R installation in another way. It does not work for Windows.
    check_call(['which', 'R'])
except CalledProcessError:
    logger.warning('R installation is needed. Multiple structure learning '
                   'methods methods use it.')
else:
    from rpy2.robjects.packages import isinstalled

    from .cl import CL
    from .fast_iamb import FastIamb
    from .grow_shrink import GrowShrink
    from .hc_tabu import HcTabu
    from .hill_climbing import HillClimbing
    from .hiton_pc import HitonPC
    from .iamb import Iamb
    from .inter_iamb import InterIamb
    from .mmhc import MMHC
    from .mmpc import MMPC
    from .naive_bayes import NB
    from .pc import PC
    from .sparsebn import SparseBn
    from .tan import Tan

    __all__ += [
        'CL',
        'FastIamb',
        'GrowShrink',
        'HillClimbing',
        'HcTabu',
        'HitonPC',
        'Iamb',
        'InterIamb',
        'MMHC',
        'MMPC',
        'NB',
        'PC',
        'SparseBn',
        'Tan',
    ]

    packnames = ('sparsebn', 'bnlearn')

    for package in packnames:
        if not isinstalled(package):
            logger.warning(
                f'R library {package} is not installed and some structure '
                f'learning methods rely on it. Please, install it!')
