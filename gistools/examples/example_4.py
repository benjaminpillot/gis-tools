# -*- coding: utf-8 -*-

""" Split polygon into sub-polygons with equal area using graph partition (requires METIS packages)

More detailed description.
"""
import numpy as np
from matplotlib import pyplot as plt

from gistools.layer import PolygonLayer


test = PolygonLayer("enp_pn_s_973.shp")
test = test[[0]].to_crs(32622)
test = test.partition(50000000, disaggregation_factor=20, precision=100, split_method="hexana",
                      contig=True, ncuts=2)
test["attr"] = np.random.randint(1000, size=(len(test),))

# Plot the resulting sub-polygons
test.plot(attribute="attr")
plt.show()

# Show corresponding areas
print(test.area)
