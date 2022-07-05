# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from .gaussian_joint import GaussianJPD
from .discrete_joint import DiscreteJPD
from .joint_distribution import JPD

__all__ = [
    'GaussianJPD',
    'DiscreteJPD',
    'JPD'
]
