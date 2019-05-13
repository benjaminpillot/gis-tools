# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
import os
from tempfile import mkstemp, gettempdir
from utils.check.type import isfile

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


class LayerTempFile:
    """ Create temp file

    """

    _file_id = 0

    def __init__(self, suffix):
        type(self)._file_id += 1
        self.path = os.path.join(gettempdir(), "tmp_layer_%d%s" % (self._file_id, suffix))
        self._files = [self.path]

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __del__(self):
        for file in self._files:
            if isfile(file):
                os.remove(file)


class ShapeTempFile(LayerTempFile):
    """ Create temporary shapefile

    """

    def __init__(self):
        super().__init__(".shp")
        self._files.extend([self.path.replace(".shp", ".shx"), self.path.replace(".shp", ".dbf")])


class GeoJSonTempFile(LayerTempFile):

    def __init__(self):
        super().__init__(".geojson")


class RasterTempFile:
    """ Create temporary raster file

    Class for creating temporary raster files used with gdal-like utilities
    """

    def __init__(self):
        self.path = mkstemp(suffix='.tif')[1]

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __del__(self):
        if isfile(self.path):
            os.remove(self.path)
