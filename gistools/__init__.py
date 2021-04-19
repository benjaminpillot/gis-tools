# -*- coding: utf-8 -*-

""" A set of geographical tools for Python developers.

GisTools is a small Python library for performing
geographical computations. Typically, it gathers
different tools from well-known libraries such as
gdal, rasterio, geopandas, fiona and shapely.
It allows easily mixing operations between vectors
and raster maps (multi-band raster are not supported
at the moment).

GisTools allows some of the following operations:
- Fast polygon intersection and split
- Polygon partition based on graph theory
(requires [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) package)
- Basic networking (shortest path)
- Download DEM from online databases
- Download OSM layers through Overpass API
- Download layers from postgis spatial database
- Extract raster statistics with respect to vector layers (polygon/line)
- Raster to/from polygons conversion
- Compute horizon obstruction from DEM

"""

# __all__ = []

__version__ = '0.17.8'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2020, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'
__licence__ = "MIT"

# from gistools.layer import *
# from gistools.raster import *
# from gistools.network import *
# from gistools.stats import *
