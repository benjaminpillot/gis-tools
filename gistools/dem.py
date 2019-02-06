# -*- coding: utf-8 -*-

""" DEM toolset (computation of slope, aspect, etc.)

More detailed description.
"""

# import

# __all__ = []
# __version__ = '0.1'
from tempfile import NamedTemporaryFile

import rasterio
from matplotlib import pyplot

from osgeo import gdal

from gistools.conversion import raster_to_array

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def compute_slope(dem):
    out_raster = NamedTemporaryFile(suffix='.tif')
    dem_options = gdal.DEMProcessingOptions(slopeFormat='percent')
    gdal.DEMProcessing(out_raster.name, dem, 'slope', options=dem_options)
    with rasterio.open(out_raster.name) as dataset:
        slope = dataset.read(1)
    return slope


if __name__ == "__main__":
    dem_test = "/home/benjamin/ownCloud/Post-doc Guyane/Data/DEM/srtm_26_11_12.tif"
    dem = raster_to_array(dem_test)
    slope_test = compute_slope(dem_test)
    print(type(slope_test))
    print(slope_test.shape)
    print(slope_test.dtype)
    slope_test[slope_test < 0] = 0
    pyplot.imshow(slope_test)
    pyplot.show()
