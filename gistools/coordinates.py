# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import math
import os

import pyproj

import numpy as np
import warnings
import geopandas as gpd
from cpc.geogrids import Geogrid
from osgeo import gdal
from rtree import index

from gistools.projections import crs_from_raster
from gistools.exceptions import GeoGridError, GeoGridWarning
from gistools.utils.check.descriptor import protected_property
from gistools.utils.check.type import type_assert, check_type, is_iterable
from gistools.utils.check.value import check_string, check_file
from gistools.utils.sys.reader import read_hdr

REF_ELLIPSOID = pyproj.pj_ellps

TIME_ZONE = {-12, -11, -10, -9.5, -9, -8, -7, -6, -5, -4, -3.5, -3, -2, -1, 0, 1, 2, 3, 3.5, 4,
             4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 8.75, 9, 9.5, 10, 10.5, 11, 12, 12.75, 13, 14}

# Be sure numpy return an error when dividing by zero
np.seterr(divide='raise')


class Ellipsoid:
    """ Ellipsoid class instance

    Store parameters of a specified ellipsoid model

    """

    def __init__(self, ellipsoid_model: str):

        ellipsoid_model = check_string(ellipsoid_model, list(REF_ELLIPSOID.keys()))
        self._f = 1 / REF_ELLIPSOID[ellipsoid_model]['rf']
        self._a = REF_ELLIPSOID[ellipsoid_model]['a']
        self._b = self._a - self._a * self._f
        self._e = math.sqrt(1 - (self._b ** 2 / self._a ** 2))
        self._model = ellipsoid_model

    @property
    def f(self):
        return self._f

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def e(self):
        return self._e

    @property
    def model(self):
        return self._model


class GeoGrid(Geogrid):
    """GeoGrid class instance

    Define regular georeferenced grid based on
    cpc.geogrids module from NOAA
    """
    latitude = protected_property('latitude')
    longitude = protected_property('longitude')
    geo_transform = protected_property('geo_transform')

    def __init__(self, ll_corner, ur_corner, res, geo_type="latlon"):
        """ GeoGrid class constructor

        See super class cpc.geogrids.Geogrid
        :param ll_corner:
        :param ur_corner:
        :param res:
        :param geo_type: grid type ('latlon' or 'equal')
        """
        super().__init__(ll_corner=ll_corner, ur_corner=ur_corner, res=res, type=geo_type)
        if self.lats == [] or self.lons == []:
            raise GeoGridError("GeoGrid has not been rightly defined (empty lats/lons field)")

        # Define geo transform used in raster computing (GDAL syntax)
        self._geo_transform = (self.ll_corner[1] - self.res/2, self.res, 0, self.ur_corner[0] + self.res/2, 0,
                               -self.res)

        # Set lat and lon numpy arrays
        self._lat = np.asarray(self.lats)[::-1]
        self._lon = np.asarray(self.lons)

        # WARNING: sorry guys, but Geogrid implementation of num_x and num_y does not seem robust to me
        # Better to define num_x and num_y as length of lon and lat rather than doing some calculation that fails due
        #  to float precision... (for really small resolutions though)
        self.num_x = len(self._lon)
        self.num_y = len(self._lat)

        # Compute lat/lon meshgrid of pixel centres
        self._longitude, self._latitude = np.meshgrid(self._lon, self._lat)

    @type_assert(lat=(int, float), lon=(int, float))
    def latlon_to_2d_index(self, lat, lon):
        if lat > self._lat.max() + self.res/2 or lat < self._lat.min() - self.res/2 or lon > self._lon.max() + \
                self.res/2 or lon < self._lon.min() - self.res/2:
            raise GeoGridError("Point is out of the geo grid")
        return np.argmin(np.abs(self._lat - lat)), np.argmin(np.abs(self._lon - lon))

    def to_res(self, new_res):
        """ Define geo grid with new resolution

        Return new GeoGrid whose resolution is different
        :param new_res:
        :return:
        """
        new_ll_corner = (self.ll_corner[0] - self.res/2 + new_res/2, self.geo_transform[0] + new_res/2)
        new_ur_corner = (self.geo_transform[3] - new_res/2, self.ur_corner[1] + self.res/2 - new_res/2)

        return GeoGrid(new_ll_corner, new_ur_corner, new_res, geo_type=self.type)

    def __eq__(self, other):

        if other.__class__ == self.__class__:
            return self.ll_corner == other.ll_corner and self.ur_corner == other.ur_corner and self.res == other.res
        else:
            return False

    def __getitem__(self, key):
        """ Get item in GeoGrid instance

        Get item in GeoGrid instance and return GeoGrid (array)
        or latitude and longitude (point)
        :param key:
        :return:
        """

        # If key is only one slice, regard it as the same slice for rows and columns
        if key.__class__ == slice:
            key = (key, key)

        new_res = self.res

        if key.__class__ == tuple:
            if key[0].__class__ == slice or key[1].__class__ == slice:
                try:
                    lat_south, lat_north = self._lat[key[0]][-1], self._lat[key[0]][0]
                except TypeError:
                    lat_south = lat_north = self._lat[key[0]]
                try:
                    lon_west, lon_east = self._lon[key[1]][0], self._lon[key[1]][-1]
                except TypeError:
                    lon_west = lon_east = self._lon[key[1]]
                new_ll_corner, new_ur_corner = (lat_south, lon_west), (lat_north, lon_east)
                return GeoGrid(new_ll_corner, new_ur_corner, new_res, self.type)
            elif key[0].__class__ == int and key[1].__class__ == int:
                return self.latitude.__getitem__(key), self.longitude.__getitem__(key)
            else:
                raise GeoGridError("Invalid indexing")
        elif key.__class__ == int:
            return self.latitude.__getitem__(key), self.longitude.__getitem__(key)
        else:
            raise GeoGridError("Invalid indexing")

    @staticmethod
    def from_geo_file(geo_file: str, res, buffer_accuracy=1, to_crs: str = ''):
        """ Get geo grid from existing geo file (such as shapefile)

        :param geo_file: path to geo file (e.g. shapefile)
        :param res: resolution of the grid
        :param buffer_accuracy: accuracy of the buffer surrounded the region (in degrees)
        :param to_crs: final coordinate reference system of the geo grid (empty string for keeping the original CRS,
        under the form 'epsg:number' or pyproj name otherwise)
        :return: GeoGrid
        """
        # Import locally (to avoid conflicts)
        from gistools.projections import proj4_from

        check_type(geo_file, str, res, (int, float), buffer_accuracy, (int, float), to_crs, str)

        if os.path.isfile(geo_file):
            geo_ds = gpd.read_file(geo_file)
        else:
            raise GeoGridError("{} is not a valid geo file".format(geo_file))

        try:
            to_crs = pyproj.CRS(to_crs)
        except ValueError:
            warnings.warn("Empty or invalid CRS/proj name. Geo grid matching geo file projection", GeoGridWarning)
        else:
            geo_ds = geo_ds.to_crs(to_crs)

        return GeoGrid.from_geopandas(geo_ds, res, buffer_accuracy)

    @staticmethod
    def from_geopandas(geopandas_series, res, buffer_accuracy=1):
        """ Get geo grid from specific geopandas database

        :param geopandas_series:
        :param res:
        :param buffer_accuracy:
        :return:
        """
        check_type(geopandas_series, (gpd.GeoDataFrame, gpd.GeoSeries),
                   res, (int, float), buffer_accuracy, (int, float))
        # Default type for geo grid
        geo_type = 'latlon'

        try:
            # If resolution in km/m, type = 'equal'
            if not pyproj.Proj(geopandas_series.crs).crs.is_geographic:
                geo_type = 'equal'
        except AttributeError:
            raise ValueError("Input GeoSeries does not seem to be valid")

        try:
            bounds = geopandas_series.bounds
        except ValueError as e:
            raise GeoGridError("GeoSeries has invalid bounds:\n {}".format(e))

        try:
            ll_corner = (buffer_accuracy * math.floor((bounds.miny.min() + res/2)/buffer_accuracy),
                         buffer_accuracy * math.floor((bounds.minx.min() + res/2)/buffer_accuracy))
            ur_corner = (buffer_accuracy * math.ceil((bounds.maxy.max() - res/2)/buffer_accuracy),
                         buffer_accuracy * math.ceil((bounds.maxx.max() - res/2)/buffer_accuracy))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            raise ValueError("Value of buffer accuracy ({}) is not valid".format(buffer_accuracy))
        else:
            return GeoGrid(ll_corner, ur_corner, res, geo_type)

    @staticmethod
    def from_raster_file(raster_file: str):
        """ Retrieve geo grid from raster file

        :param raster_file: path to raster file
        :return: GeoGrid
        """
        check_type(raster_file, str)

        try:
            source_ds = gdal.Open(raster_file)
        except RuntimeError as error:
            raise error

        geo_transform = source_ds.GetGeoTransform()

        # Warning: ll_corner and ur_corner correspond to grid corners
        # that is pixel centers, but gdal rasters are defined with respect
        # to pixel corners... That's why "+- res/2"
        ll_corner = (geo_transform[3] + source_ds.RasterYSize * geo_transform[5] - geo_transform[5] / 2, geo_transform[
                     0] + geo_transform[1] / 2)
        ur_corner = (geo_transform[3] + geo_transform[5] / 2, geo_transform[0] + source_ds.RasterXSize *
                     geo_transform[1] - geo_transform[1] / 2)

        # Geo type
        if crs_from_raster(raster_file).is_geographic:
            geo_type = "latlon"
        else:
            geo_type = "equal"

        if math.isclose(math.fabs(geo_transform[1]), math.fabs(geo_transform[5])):
            geo_grid = GeoGrid(ll_corner, ur_corner, math.fabs(geo_transform[1]), geo_type)
        else:
            geo_grid = None
            warnings.warn('No regular raster: no GeoGrid implemented', Warning)

        return geo_grid

    @staticmethod
    def from_hdr(hdr_file: str):
        """ Get geo grid from envi geospatial header file

        :param hdr_file:
        :return:
        """
        check_file(hdr_file, '.hdr')
        hdr_info = read_hdr(hdr_file)
        if hdr_info['y_res'] != hdr_info['x_res']:
            raise AttributeError("X and Y image resolution must be the same")

        res = hdr_info['y_res']
        ll_corner = (hdr_info['y_origin'] - hdr_info['y_size'] * res + res/2, hdr_info['x_origin'] + res/2)
        ur_corner = (hdr_info['y_origin'] - res/2, hdr_info['x_origin'] + hdr_info['x_size'] * res - res/2)

        if "Lat/Lon" in hdr_info["proj"]:
            geotype = "latlon"
        else:
            geotype = "equal"

        return GeoGrid(ll_corner, ur_corner, res, geotype)


def r_tree_idx(geometry_collection):
    """ Return Rtree spatial index of geometry collection

    :param geometry_collection: geometry collection (list, series, etc.)
    :return:
    """

    idx = index.Index()
    if not is_iterable(geometry_collection):
        geometry_collection = [geometry_collection]

    for i, geom in enumerate(geometry_collection):
        idx.insert(i, geom.bounds)

    return idx
