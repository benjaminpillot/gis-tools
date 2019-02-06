# -*- coding: utf-8 -*-

""" Compute horizon obstruction from DEM


"""

from matplotlib import pyplot as plt

from gistools.topography import get_horizon
from gistools.raster import DigitalElevationModel
from gistools.utils.sys.timer import Timer

__version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


_dem = DigitalElevationModel("srtm_38_04.tif", no_data_value=-32768)
_dem = _dem.to_crs(4326)
with Timer() as t:
    test = get_horizon(42, 9, _dem, distance=0.2, precision=1)
print("time: %s" % t)
with Timer() as t:
    for _ in range(20):
        test = get_horizon(42, 9, _dem, distance=0.2, precision=1)
print("time: %s" % t)
plt.plot(test["azimuth"], test["elevation"])
plt.show()
