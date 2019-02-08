# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""
from matplotlib import pyplot as plt

from gistools.raster import DigitalElevationModel

__version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


test = DigitalElevationModel.from_cgiar_online_database((8, 38, 14, 42))
test.plot()
plt.show()
