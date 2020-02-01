# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point, MultiPoint

__version__ = '0.16.0'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'

# from gistools.layer import *
# from gistools.raster import *
# from gistools.network import *
# from gistools.stats import *


GEOMETRY_CLASS = {'linestring': (LineString, MultiLineString), 'polygon': (Polygon, MultiPolygon),
                  'point': (Point, MultiPoint)}
