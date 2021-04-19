# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import os
import uuid
from tempfile import mkstemp, gettempdir


class File:

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TempFile(File):

    def __del__(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


class ShapeTempFile(TempFile):

    def __init__(self):
        self.name = os.path.join(gettempdir(), str(uuid.uuid4()))
        super().__init__(self.name + ".shp")

    def __del__(self):
        super().__del__()
        for ext in [".shx", ".dbf", ".prj", ".cpg"]:
            try:
                os.remove(self.name + ext)
            except FileNotFoundError:
                pass


class RasterTempFile(TempFile):
    """ Create temporary raster file

    """
    def __init__(self):
        super().__init__(mkstemp(suffix='.tif')[1])


# class LayerTempFile:
#     """ Create temp file
#
#     """
#
#     _file_id = 0
#
#     def __init__(self, suffix):
#         type(self)._file_id += 1
#         self.path = os.path.join(gettempdir(), "tmp_layer_%d%s" % (self._file_id, suffix))
#         self._files = [self.path]
#
#     def __enter__(self):
#         return self.path
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         del self
#
#     def __del__(self):
#         for file in self._files:
#             if isfile(file):
#                 os.remove(file)
#
#
# class ShapeTempFile(LayerTempFile):
#     """ Create temporary shapefile
#
#     """
#
#     def __init__(self):
#         super().__init__(".shp")
#         self._files.extend([self.path.replace(".shp", ".shx"),
#                             self.path.replace(".shp", ".dbf"),
#                             self.path.replace(".shp", ".prj")])


# class GeoJSonTempFile(LayerTempFile):
#
#     def __init__(self):
#         super().__init__(".geojson")


class GeoJSonTempFile(TempFile):

    def __init__(self):
        self.name = os.path.join(gettempdir(), str(uuid.uuid4()))
        super().__init__(os.path.join(gettempdir(), str(uuid.uuid4())) + ".geojson")


# class RasterTempFile:
#     """ Create temporary raster file
#
#     Class for creating temporary raster files used with gdal-like utilities
#     """
#
#     def __init__(self):
#         self.path = mkstemp(suffix='.tif')[1]
#
#     def __enter__(self):
#         return self.path
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         del self
#
#     def __del__(self):
#         if isfile(self.path):
#             os.remove(self.path)
