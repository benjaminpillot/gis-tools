# -*- coding: utf-8 -*-

""" Define exceptions used by the geotools package

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


class ZonalStatisticsError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class RasterMapError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class DigitalElevationModelError(RasterMapError):
    pass


class RasterMapWarning(Warning):
    pass


class GeoLayerError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class RoadNetworkError(NetworkError):
    pass


class GeoLayerWarning(Warning):
    pass


if __name__ == "__main__":
    pass
