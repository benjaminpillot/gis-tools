# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""
import numpy as np
from matplotlib import pyplot as plt

from gistools.layer import PolygonLayer
from gistools.utils.sys.timer import Timer

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


test = PolygonLayer("enp_pn_s_973.shp")
test = test[[0]].to_crs(32622)
with Timer() as t:
    test = test.split_into_equal_areas(50000000, disaggregation_factor=20, precision=100, split_method="hexana",
                                       contig=True, ncuts=2)
print("spend time: %s" % t)
test["attr"] = np.random.randint(1000, size=(len(test),))

# Plot the resulting sub-polygons
test.plot(attribute="attr")
plt.show()

# Show corresponding areas
print(test.area)
