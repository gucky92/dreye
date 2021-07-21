"""
"""

from dreye.datasets.spitschan2016 import load_dataset as load_spitschan2016
from dreye.datasets.granada import load_dataset as load_granada
from dreye.datasets.flowers import load_dataset as load_flowers
from dreye.datasets.human import load_dataset as load_human

__all__ = [
    'load_spitschan2016',
    'load_granada',
    'load_flowers', 
    'load_human'
]
