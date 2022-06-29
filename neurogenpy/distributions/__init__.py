# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/

# Licensed under GNU General Public License v3.0:
# https://www.gnu.org/licenses/gpl-3.0.html

from .gaussian_joint import GaussianJointDistribution
from .discrete_joint import DiscreteJointDistribution

__all__ = [
    'GaussianJointDistribution',
    'DiscreteJointDistribution'
]
