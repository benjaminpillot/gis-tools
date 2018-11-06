# -*- coding: utf-8 -*-

""" Example 1

More detailed description.
"""

from gis_tools.raster import DigitalElevationModel
from gis_tools.layer import PolygonLayer

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2017, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


# Create a digital elevation model from SRTM DEM of French Guiana
dem = DigitalElevationModel("srtm_guyana.tif", no_data_value=-32768)

# Convert DEM to UTM coordinate reference system
dem_utm = dem.to_crs({'init': 'epsg:32622'})

# Compute slope from the DEM
slope = dem_utm.compute_slope()

# Import a geographical layer made of 6 large polygons from a shapefile
layer = PolygonLayer("enp_pn_s_973.shp")

# Create ZonalStatistics instance from slope and layer. 
zonal_stat = ZonalStatistics(slope, layer, is_surface_weighted=False, all_touched=True)

# Compute slope average within each polygon
avg = zonal_stat.mean()
print(avg)
