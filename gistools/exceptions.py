# -*- coding: utf-8 -*-

""" Define exceptions used by the gistools package

More detailed description.
"""

from cpc.geogrids.exceptions import GeogridError

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


class GeoGridError(GeogridError):
    pass


class GeoGridWarning(Warning):
    pass


class ProjectionWarning(Warning):
    pass


class ZonalStatisticsError(Exception):
    pass


class RasterMapError(Exception):
    pass


class DigitalElevationModelError(RasterMapError):
    pass


class RasterMapWarning(Warning):
    pass


class GeoLayerError(Exception):
    pass


class GeoLayerEmptyError(GeoLayerError):
    pass


class PolygonLayerError(GeoLayerError):
    pass


class LineLayerError(GeoLayerError):
    pass


class PointLayerError(GeoLayerError):
    pass


class EdgeError(LineLayerError):
    pass


class RoadError(EdgeError):
    pass


class NodeError(PointLayerError):
    pass


class RoadNodeError(NodeError):
    pass


class NetworkError(Exception):
    pass


class RoadNetworkError(NetworkError):
    pass


class SpatialDatabaseError(Exception):
    pass


class GeoLayerWarning(Warning):
    pass


class PolygonLayerWarning(GeoLayerWarning):
    pass


class ImportMetisWarning(Warning):
    pass


class SpatialDatabaseWarning(Warning):
    pass


class QlQueryError(Exception):
    pass


class AddressConverterError(Exception):
    pass


class DictionaryConverterError(AddressConverterError):
    pass
