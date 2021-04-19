# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""
from matplotlib import pyplot as plt

from gistools.raster import DigitalElevationModel


test = DigitalElevationModel.from_cgiar_online_database((8, 38, 14, 42))
test.plot()
plt.show()
