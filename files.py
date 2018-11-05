# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# import

# __all__ = []
# __version__ = '0.1'
import os
from tempfile import mkstemp, mktemp

from utils.check import isfile

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


class ShapeTempFile:
    """ Create temporary shapefile

    """

    def __init__(self):
        self.path = mktemp(suffix=".shp")
        self._files = [self.path, self.path.replace(".shp", ".shx"), self.path.replace(".shp", ".dbf")]

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __del__(self):
        for file in self._files:
            if isfile(file):
                os.remove(file)


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


if __name__ == "__main__":
    pass
