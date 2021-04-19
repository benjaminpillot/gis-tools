# -*- coding: utf-8 -*-

""" Compute horizon obstruction from DEM


"""

from matplotlib import pyplot as plt

from gistools.topography import get_horizon
from gistools.raster import DigitalElevationModel


_dem = DigitalElevationModel("srtm_38_04.tif", no_data_value=-32768)
_dem = _dem.to_crs(4326)
test = get_horizon(42, 9, _dem, distance=0.2, precision=1)
plt.plot(test["azimuth"], test["elevation"])
plt.show()
