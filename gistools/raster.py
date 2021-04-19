# -*- coding: utf-8 -*-

""" Raster map classes and methods

Toolset for working with static raster/array maps,
defined as matrices of cell values lying on a
geo-referenced grid
"""
import uuid

from matplotlib import pyplot as plt
from rasterio import open as rasterio_open
from rasterio.merge import merge as rasterio_merge
from urllib.error import URLError
from urllib.request import urlretrieve
from zipfile import ZipFile

import pyproj
import warnings
import os
import tempfile
from functools import wraps
import copy

import numpy as np
from osgeo import gdal, ogr

from gistools.coordinates import GeoGrid
from gistools.conversion import raster_to_array, array_to_raster
from gistools.exceptions import RasterMapError, GeoGridError, DigitalElevationModelError
from gistools.files import RasterTempFile, ShapeTempFile
from gistools.layer import PolygonLayer, check_proj
from gistools.projections import wkt_from, srs_from, \
    ellipsoid_from, crs_from_raster
from gistools.surface import compute_surface
from gistools.utils.check.descriptor import protected_property
from gistools.utils.check.type import check_type, type_assert, collection_type_assert, isfile
from gistools.utils.check.value import check_string

gdal.UseExceptions()


CGIAR_URL = "http://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
DEFAULT_OUTPUT = os.path.join(tempfile.gettempdir(), "out.tif")


# Decorator for returning new instance of RasterMap
def return_new_instance(method):
    @wraps(method)
    def _return_new_instance(self, *args, **kwargs):
        output = method(self, *args, **kwargs)
        if isinstance(output, np.ndarray):
            new_self = self.__class__(output, self.geo_grid, no_data_value=self.no_data_value, crs=self.crs)
        elif isinstance(output, tuple):
            new_self = self.__class__(output[0], output[1],
                                      no_data_value=self.no_data_value,
                                      crs=self.crs)  # Return
            # array AND GeoGrid
        else:
            new_self = output
        return new_self
    return _return_new_instance


def gdal_decorator(no_data_value=None):
    def decorate(method):
        no_data = no_data_value

        @wraps(method)
        def _gdal_decorator(self, *args, **kwargs):
            with RasterTempFile() as out_raster:
                method(self, out_raster, *args, **kwargs)
                if no_data is None:
                    return self.__class__(out_raster, no_data_value=self.no_data_value)
                else:
                    return self.__class__(out_raster, no_data_value=no_data)
        return _gdal_decorator
    return decorate


class GdalOpen:

    def __init__(self, raster_file):
        self.dataset = gdal.Open(raster_file)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        # del self.dataset
        self.dataset = None

# TODO: raster map with band number


class RasterMap:
    """ RasterMap base class

    A RasterMap is a numpy array corresponding to a geo raster
    When the raster is stored using numpy array, and when it is
    possible, methods rely on numpy/rasterio packages, otherwise
    gdal is used
    """

    # Mapping numpy types to gdal types (thanks to
    # https://gist.github.com/CMCDragonkai/ac6289fa84bcc8888035744d7e00e2e6)
    _numpy_to_gdal_type = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 3, 'uint32': 4, 'int32': 5, 'float32': 6,
                           'float64': 7, 'complex64': 10, 'compex128': 11}

    _numpy_to_ogr_type = {'uint8': 0, 'int8': 0, 'uint16': 0, 'int16': 0, 'uint32': 0, 'int32': 0, 'float32': 2,
                          'float64': 2}

    # GDAL/OGR underlying attributes
    _gdal_drv = gdal.GetDriverByName('GTiff')
    _ogr_shp_drv = ogr.GetDriverByName('ESRI Shapefile')
    _ogr_geojson_drv = ogr.GetDriverByName('GeoJSON')

    # Temp file
    _temp_raster_file = None

    # SetAccess = protected (property decorator)
    geo_grid = protected_property('geo_grid')
    raster_array = protected_property('raster_array')
    crs = protected_property('crs')
    x_origin = protected_property('x_origin')
    y_origin = protected_property('y_origin')
    res = protected_property('res')
    x_size = protected_property('x_size')
    y_size = protected_property('y_size')
    shape = protected_property('shape')
    no_data_value = protected_property('no_data_value')

    def __init__(self, raster, geo_grid: GeoGrid = None, no_data_value=None, crs=None):
        """ RasterMap constructor

        :param raster: raster file or numpy array
        :param geo_grid: GeoGrid instance
        :param no_data_value: data to be regarded as "no_data"
        :param crs: projection used for the raster

        :Example:
        >>>
        """
        check_type(raster, (str, np.ndarray))

        if type(raster) == np.ndarray:
            raster_file = None
            try:
                test_fit = geo_grid.data_fits(raster)
                crs = pyproj.CRS(crs)
                # crs = proj4_from(crs)
                if not test_fit:
                    raise RasterMapError("Input geo grid does not fit raster")
            except AttributeError:
                raise RasterMapError("Geo grid argument has not been set")
            except (ValueError, TypeError):
                raise RasterMapError("Invalid projection: crs='{}'".format(crs))
        else:
            raster_file = raster
            try:
                geo_grid = GeoGrid.from_raster_file(raster)
                crs = crs_from_raster(raster)
                raster = raster_to_array(raster)
            except RuntimeError:
                raise RasterMapError("Invalid/unknown file '%s'" % raster_file)

        # Set attributes
        self._raster_file = raster_file
        self._geo_grid = geo_grid
        self._raster_array = np.array(raster, dtype='float64')  # Ensure compatibility with NaNs
        self._crs = crs
        self._res = self._geo_grid.res
        self._x_origin = self._geo_grid.geo_transform[0]
        self._y_origin = self._geo_grid.geo_transform[3]
        self._x_size = self._geo_grid.num_x
        self._y_size = self._geo_grid.num_y
        self._shape = self._raster_array.shape
        self._no_data_value = no_data_value

        if no_data_value is not None:
            self._raster_array[self._raster_array == no_data_value] = np.nan  # Use attribute (raster_array) rather than
            # instance (self == ...) to avoid 'recursion' error with decorator above

        # Available filters
        self._filters = {"majority_filter": self._majority_filter, "sieve": self._gdal_sieve}

    def __del__(self):
        try:
            os.remove(self._temp_raster_file)
        except TypeError:
            pass

    @type_assert(filter_name=str)
    def apply_filter(self, filter_name, *args, **kwargs):
        """ Apply filter to raster

        :param filter_name:
        :param args: list of arguments related to filter function
        :param kwargs: list of keyword args related to filter function
        :return:
        """
        return self._filters[filter_name](*args, **kwargs)

    @return_new_instance
    @type_assert(geo_layer=PolygonLayer)
    def clip(self, geo_layer: PolygonLayer, crop=False, all_touched=False):
        """ Clip raster according to GeoLayer polygon(s)

        Keep only points which are inside polygon boundaries
        and crop raster if necessary.
        :param geo_layer: GeoLayer instance
        :param crop: if True, crop raster
        :param all_touched: when True, all cells touched by polygons are considered within
        :return:
        """
        check_proj(geo_layer.crs, self.crs)
        if crop:
            new_raster = self.get_raster_at(geo_layer)
            return new_raster.clip(geo_layer, crop=False, all_touched=all_touched)
        else:
            return self._burn_layer_values(geo_layer, False, all_touched)

    @return_new_instance
    def contour(self, interval, absolute_interval=True, percentile_min=2, percentile_max=98):
        """ Extract contour from raster

        :param interval: interval between contour lines
        :param absolute_interval: relative or absolute interval ?
        :param percentile_min:
        :param percentile_max:
        :return:
        """
        values = self.raster_array[~np.isnan(self.raster_array)]
        v_min, v_max = np.percentile(values, [percentile_min, percentile_max])
        if not absolute_interval:
            interval *= v_max
        contour_range = np.linspace(v_min, v_max, int((v_max - v_min)/interval) + 1)
        new_raster_array = np.zeros(self.raster_array.shape)
        new_raster_array[self.raster_array_without_nans < v_min] = np.mean(values[values < v_min])
        new_raster_array[self.raster_array_without_nans >= v_max] = np.mean(values[values >= v_max])

        for bins in zip(contour_range[0:-1], contour_range[1::]):
            new_raster_array[(self.raster_array_without_nans >= bins[0]) & (self.raster_array_without_nans < bins[
                1])] = np.mean(values[(values >= bins[0]) & (values < bins[1])])

        new_raster_array[self.is_no_data()] = self.no_data_value

        return new_raster_array

    def copy(self):
        return copy.deepcopy(self)

    @return_new_instance
    @type_assert(factor=int, method=str, no_limit=bool)
    def disaggregate(self, factor: int = 1, method: str = 'nearest', no_limit=False):
        """ Disaggregate raster cells

        :param factor: scale factor for disaggregation (number of cells)
        :param method: 'linear' or 'nearest' (default = 'nearest')
        :param no_limit: no limit for disaggregation (default=False)
        :return:
        """
        from scipy.interpolate import RegularGridInterpolator

        upper_limit = np.inf if no_limit else 10**8 / (self.geo_grid.num_x * self.geo_grid.num_y)
        if 1 < factor <= upper_limit:
            new_geo_grid = self.geo_grid.to_res(self.res/factor)
            try:
                interpolator = RegularGridInterpolator((self.geo_grid.lats, self.geo_grid.lons),
                                                       self.raster_array[::-1, :], bounds_error=False,
                                                       method=method)
            except ValueError:
                raise RasterMapError("Method should be 'linear' or 'nearest' but is {}".format(method))

            return interpolator((new_geo_grid.latitude, new_geo_grid.longitude)), new_geo_grid
        else:
            warnings.warn("Invalid factor, factor = 1, or exceeded limit (set no_limit=True). Return copy of object")
            return self.copy()

    @return_new_instance
    @type_assert(geo_layer=PolygonLayer)
    def exclude(self, geo_layer: PolygonLayer, all_touched=False):
        """ Exclude raster cells within GeoLayer polygon(s)

        Keep only points outside from layer boundaries
        :param geo_layer:
        :param all_touched: when True, all cells touched by polygons are regarded as within
        :return:
        """
        check_proj(geo_layer.crs, self.crs)
        return self._burn_layer_values(geo_layer, True, all_touched)

    def gdal_clip(self, extent):
        """ Static method for clipping large raster

        :param extent: array/list as [x_min, y_min, x_max, y_max]
        :return:
        """
        pass

    def gdal_resample(self, factor):
        """ Resampling raster using GDAL

        Unlike the 'disaggregate' method, GDAL uses
        files for resampling (faster for large areas)
        :param factor:

        :return:
        """
        return self._gdal_resample_raster(factor)

    @return_new_instance
    @collection_type_assert(ll_point=dict(collection=(list, tuple), length=2, type=(int, float)),
                            ur_point=dict(collection=(list, tuple), type=(int, float), length=2))
    def get_raster_at(self, layer=None, ll_point=None, ur_point=None):
        """ Extract sub-raster in current raster map

        Extract new raster from current raster map
        by giving either a geo lines_ or a new geo-square
        defined by lower-left point (ll_point) and upper
        right point (ur_point) such as for geo grids.
        :param layer: GeoLayer instance
        :param ll_point: tuple of 2 values (lat, lon)
        :param ur_point: tuple of 2 values (lat, lon)
        :return: RasterMap
        """
        if layer is not None:
            check_proj(layer.crs, self.crs)
            ll_point = (layer.bounds[1], layer.bounds[0])  # Warning: (lat, lon) in that order !
            ur_point = (layer.bounds[3], layer.bounds[2])

        try:
            ll_point_r, ll_point_c = self._geo_grid.latlon_to_2d_index(ll_point[0], ll_point[1])
            ur_point_r, ur_point_c = self._geo_grid.latlon_to_2d_index(ur_point[0], ur_point[1])
            return self.raster_array[ur_point_r:ll_point_r + 1, ll_point_c:ur_point_c + 1], \
                self.geo_grid[ur_point_r:ll_point_r + 1, ll_point_c:ur_point_c + 1]
        except GeoGridError:
            raise RasterMapError("Lower left or/and upper right points have not been rightly defined")

    def get_value_at(self, latitude, longitude):
        """ Get single raster value at latitude, longitude

        :param latitude:
        :param longitude:
        :return:
        """
        r, c = self._geo_grid.latlon_to_2d_index(latitude, longitude)
        return self._raster_array[r, c]

    def is_latlong(self):
        return self.crs.is_geographic

    def is_no_data(self):
        return np.isnan(self.raster_array)

    def max(self):
        """ Return maximum value of raster

        :return:
        """
        return np.nanmax(self.raster_array)

    def mean(self):
        """ Compute mean of raster map

        Mean of raster values
        :return:
        """
        return np.nanmean(self.raster_array)

    def min(self):
        """ Return minimum value of raster

        :return:
        """
        return np.nanmin(self.raster_array)

    def plot(self, ax=None, cmap=None, colorbar=False, colorbar_title=None, **kwargs):
        """ Plot Raster

        :param ax:
        :param cmap:
        :param colorbar:
        :param colorbar_title:
        :param kwargs:
        :return:
        """

        extent = [self.x_origin, self.x_origin + self.res * self.x_size,
                  self.y_origin - self.res * self.y_size, self.y_origin]

        if ax is None:
            _, ax = plt.subplots()

        # Use imshow to plot raster
        img = ax.imshow(self.raster_array, extent=extent, cmap=cmap,
                        vmin=self.min(), vmax=self.max(), **kwargs)

        if colorbar:
            cbar = plt.colorbar(img)
            cbar.ax.set_ylabel(colorbar_title)

        return ax

    @type_assert(field_name=str)
    def polygonize(self, field_name, layer_name="layer", is_8_connected=False):
        """ Convert raster into vector polygon(s)

        :param field_name: name of the corresponding field in the final shape file
        :param layer_name: name of resulting layer
        :param is_8_connected: pixel connectivity used for polygon
        :return:
        """
        check_type(is_8_connected, bool)
        with ShapeTempFile() as out_shp:
            self._gdal_polygonize(out_shp, layer_name, field_name, is_8_connected)
            return PolygonLayer(out_shp, name=layer_name)

    def sum(self):
        """ Compute sum of raster map

        :return:
        """
        return np.nansum(self.raster_array)

    def surface(self):
        """ Return array of raster cell surface values

        :return: numpy ndarray
        """
        surface = compute_surface(self.geo_grid.longitude - self.geo_grid.res / 2, self.geo_grid.longitude +
                                  self.geo_grid.res / 2, self.geo_grid.latitude + self.geo_grid.res / 2,
                                  self.geo_grid.latitude - self.geo_grid.res / 2, self.geo_type,
                                  ellipsoid_from(self.crs))

        return surface

    def to_crs(self, crs):
        """ Reproject raster onto new CRS

        :param crs:
        :return:
        """
        try:
            if self.crs != crs:
                srs = srs_from(crs)
                return self._gdal_warp(srs)
            else:
                return self.copy()
        except ValueError:
            raise RasterMapError("Invalid CRS '%s'" % crs)

    def to_file(self, raster_file, data_type=None):
        """ Save raster to file

        :param raster_file:
        :param data_type: data type
        :return:
        """
        if data_type is None:
            dtype = self._numpy_to_gdal_type[self.data_type]
        else:
            try:
                dtype = self._numpy_to_gdal_type[data_type]
            except KeyError:
                raise RasterMapError("Invalid data type '%s'" % data_type)

        out = array_to_raster(raster_file, self.raster_array_without_nans,
                              self.geo_grid, self.crs, datatype=dtype,
                              no_data_value=self.no_data_value)

        return out

    @classmethod
    def is_equally_referenced(cls, *args):
        """ Check geo referencing equality between rasters

        :param args:
        :return:
        """
        return False not in [raster_1.geo_grid == raster_2.geo_grid for raster_1, raster_2 in zip(args[:-1], args[1::])]

    @property
    def data_type(self):
        return str(self.raster_array.dtype)

    @property
    def geo_type(self):
        if self.is_latlong():
            return 'latlon'
        else:
            return 'equal'

    @property
    def raster_array_without_nans(self):
        array = self.raster_array.copy()
        array[np.isnan(array)] = self.no_data_value
        return array

    @property
    def raster_file(self):
        """ Return underlying raster file

        If raster file does not exist, create a temporary one
        :return:
        """
        if not isfile(self._raster_file):
            self._raster_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            self.to_file(self._raster_file)
            self._temp_raster_file = self._raster_file
            # with RasterTempFile() as file:
            #     self._raster_file = file
            # self.to_file(self._raster_file)

        return self._raster_file

    @classmethod
    def merge(cls, list_of_raster, bounds=None):
        """ Merge several RasterMap instances

        :param list_of_raster: list of RasterMap class instance(s)
        :param bounds: bounds of the output RasterMap
        :return:
        """
        # TODO: use gdal merge method
        list_of_datasets = []
        for raster_map in list_of_raster:
            list_of_datasets.append(rasterio_open(raster_map.raster_file, 'r'))

        # Merge using rasterio
        array, transform = rasterio_merge(list_of_datasets, bounds=bounds)
        with RasterTempFile() as file:
            with rasterio_open(file, 'w', driver="GTiff", height=array.shape[1], width=array.shape[2], count=1,
                               dtype=array.dtype, crs=list_of_raster[0].crs, transform=transform) as out_dst:
                out_dst.write(array.squeeze(), 1)
            return cls(file, no_data_value=list_of_raster[0].no_data_value)

    def __getitem__(self, key):

        if key.__class__ == type(self):
            key = key.raster_array

        if key.__class__ == slice:
            key = (key, key)

        if key.__class__ == tuple:
            if key[0].__class__ == int and key[1].__class__ == int:
                return self.raster_array.__getitem__(key)
            else:
                try:
                    return self.__class__(self.raster_array.__getitem__(key), self.geo_grid.__getitem__(key),
                                          no_data_value=self.no_data_value, crs=self.crs)
                except IndexError:
                    raise RasterMapError("Invalid indexing")
                except Exception as e:
                    raise RuntimeError("Unknown error while getting data in raster map: {}".format(e))
        elif key.__class__ == np.ndarray:
            return self.raster_array.__getitem__(key)
        else:
            raise RasterMapError("Invalid indexing")

    def __setitem__(self, key, value):

        if type(key) == type(self):
            self.raster_array.__setitem__(key.raster_array, value)
        else:
            try:
                self.raster_array.__setitem__(key, value)
            except IndexError:
                raise RasterMapError("Invalid indexing")
            except Exception as e:
                raise RuntimeError("Unknown error while setting data in raster map: {}".format(e))

        return self

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return self._apply_operator(other, '__mul__', '*')

    def __add__(self, other):
        return self._apply_operator(other, '__add__', '+')

    def __sub__(self, other):
        return self._apply_operator(other, '__sub__', '-')

    def __and__(self, other):
        return self._apply_comparison(other, '__and__', '&')

    def __or__(self, other):
        return self._apply_comparison(other, '__or__', '|')

    def __xor__(self, other):
        return self._apply_comparison(other, '__xor__', '^')

    def __eq__(self, other):
        return self._apply_comparison(other, '__eq__', '==')

    def __ne__(self, other):
        return self._apply_comparison(other, '__ne__', '!=')

    def __gt__(self, other):
        return self._apply_comparison(other, '__gt__', '>')

    def __lt__(self, other):
        return self._apply_comparison(other, '__lt__', '<')

    def __ge__(self, other):
        return self._apply_comparison(other, '__ge__', '>=')

    def __le__(self, other):
        return self._apply_comparison(other, '__le__', '<=')

    @return_new_instance
    def __neg__(self):
        return -self.raster_array

    # @return_new_instance
    def _apply_comparison(self, other, operator_function, operator_str):
        valid_values = np.full(self.raster_array.shape, False)
        if isinstance(other, RasterMap):
            other = other.raster_array
        try:
            valid_values[(~np.isnan(self.raster_array)) & (~np.isnan(other))] = True
            valid_values[valid_values] = self.raster_array[valid_values].__getattribute__(operator_function)(
                other[valid_values])
            return valid_values
        except (TypeError, IndexError):
            valid_values[~np.isnan(self.raster_array)] = True
            valid_values[valid_values] = self.raster_array[valid_values].__getattribute__(operator_function)(other)
            return valid_values
        except Exception as e:
            raise RasterMapError("Comparison for '{}' has failed ({})".format(operator_str, e))

    @return_new_instance
    def _apply_operator(self, other, operator_function, operator_str):
        if isinstance(other, RasterMap):
            if self.geo_grid == other.geo_grid:
                return self.raster_array.__getattribute__(operator_function)(other.raster_array)
            else:
                raise RasterMapError("Raster maps are not defined on the same geo grid")
        else:
            try:
                return self.raster_array.__getattribute__(operator_function)(other)
            except TypeError:
                raise RasterMapError("Unsupported operand type(s) for {}: '{}' and '{}'".format(operator_str,
                                                                                                type(self).__name__,
                                                                                                type(other).__name__))
            except ValueError:
                raise RasterMapError("No match for operand {} between '{}' and '{}'".format(operator_str,
                                                                                            type(self).__name__,
                                                                                            type(other).__name__))
            except Exception as e:
                raise RuntimeError("Unexpected error when applying {} between '{}' and '{}': {}"
                                   .format(operator_str, type(self).__name__, type(other).__name__, e))

    def _burn_layer_values(self, geo_layer, mask, all_touched):
        """ Burn raster values inside or outside layer

        :param geo_layer:
        :param mask:
        :param all_touched:
        :return:
        """
        layer = geo_layer.copy()
        layer["burn_value"] = 1
        layer = layer.to_array(self.geo_grid, "burn_value", all_touched=all_touched)
        new_raster_array = self.raster_array.copy()
        if mask:
            new_raster_array[layer == 1] = np.nan
        else:
            new_raster_array[layer != 1] = np.nan

        return new_raster_array

    def _gdal_polygonize(self, out_shp, layer_name, field_name, is_8_connected):
        """ Polygonize raster using GDAL

        :param out_shp: shapefile
        :param layer_name:
        :param field_name:
        :param is_8_connected:
        :return:
        """
        connectivity = "8CONNECTED=%d" % (8 if is_8_connected else 4)

        with GdalOpen(self.raster_file) as src_ds:
            src_band = src_ds.GetRasterBand(1)
            dst_ds = self._ogr_shp_drv.CreateDataSource(out_shp)
            dst_layer = dst_ds.CreateLayer(layer_name, srs_from(self.crs))

            fd = ogr.FieldDefn(field_name, self._numpy_to_ogr_type[self.data_type])
            dst_layer.CreateField(fd)

            gdal.Polygonize(src_band, src_band.GetMaskBand(), dst_layer, 0, [connectivity])

    @gdal_decorator()
    def _gdal_resample_raster(self, out_raster, factor):
        """ Resample raster using GDAL utility

        :param factor:
        :return:
        """
        with GdalOpen(self.raster_file) as source_ds:
            dst_ds = self._gdal_drv.Create(out_raster, self.x_size * factor, self.y_size * factor, 1,
                                           source_ds.GetRasterBand(1).DataType)
            resample_geo_transform = (self.x_origin, self.res / factor, 0, self.y_origin, 0, -self.res / factor)
            dst_ds.SetGeoTransform(resample_geo_transform)
            dst_ds.SetProjection(wkt_from(self.crs))
            gdal.RegenerateOverview(source_ds.GetRasterBand(1), dst_ds.GetRasterBand(1), 'mode')

    @gdal_decorator()
    def _gdal_warp(self, output_raster, srs):
        with GdalOpen(self.raster_file) as src_ds:
            gdal.Warp(output_raster, src_ds, dstSRS=srs)

    @gdal_decorator()
    def _gdal_sieve(self, out_raster, *args, **kwargs):
        """ Sieve filter using gdal

        :param args:
        :param kwargs:
        :return:
        """
        # Destination dataset
        dst_ds = self._clone_gdal_dataset(out_raster)

        # Apply sieve filter
        with GdalOpen(self.raster_file) as source_ds:
            gdal.SieveFilter(source_ds.GetRasterBand(1), None,
                             dst_ds.GetRasterBand(1), *args, **kwargs)

    def _majority_filter(self):
        pass

    def _clone_gdal_dataset(self, out_raster):
        """ Create GDAL dataset based on raster map properties

        :param out_raster: out raster file
        :return:
        """
        with GdalOpen(self.raster_file) as source_ds:
            dst_ds = self._gdal_drv.Create(out_raster, self.x_size, self.y_size, 1,
                                           source_ds.GetRasterBand(1).DataType)
            dst_ds.SetGeoTransform(self.geo_grid.geo_transform)
            dst_ds.SetProjection(wkt_from(self.crs))

            if self.no_data_value is not None:
                dst_band = dst_ds.GetRasterBand(1)
                dst_band.SetNoDataValue(self.no_data_value)

        return dst_ds


class ClassificationMap(RasterMap):
    """ Classification raster map

    Store classification values as integers
    """

    def __init__(self, raster, geo_grid, crs):
        super().__init__(raster, geo_grid, crs=crs, no_data_value=None)

        # Update available filters
        self._filters.update({"sieve": self._gdal_sieve})

    @staticmethod
    @type_assert(raster_map=RasterMap)
    def from_raster_map(raster_map):
        """ Build classification raster from RasterMap instance

        :param raster_map:
        :return:
        """
        pass


class DigitalElevationModel(RasterMap):
    """ Digital Elevation SubModel class

    Store Digital Elevation SubModel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_slope(self, slope_format="percent"):
        """ Compute slope from DEM

        :return:
        """
        slope_format = check_string(slope_format, {'percent', 'degree'})
        return self._compute_slope(slope_format)

    def compute_aspect(self):
        """ Compute aspect

        :return:
        """
        return self._compute_aspect()

    # TODO: relate following method to functions in "topography" module
    def get_horizon(self):
        pass

    @gdal_decorator(no_data_value=-9999)
    def _compute_slope(self, out_raster, slope_format):

        with GdalOpen(self.raster_file) as src_ds:
            options = gdal.DEMProcessingOptions(slopeFormat=slope_format)
            gdal.DEMProcessing(out_raster, src_ds, 'slope', options=options)

    @gdal_decorator(no_data_value=-9999)
    def _compute_aspect(self, out_raster):

        with GdalOpen(self.raster_file) as src_ds:
            gdal.DEMProcessing(out_raster, src_ds, 'aspect')

    @staticmethod
    @type_assert(bounds=tuple, product=str, margin=(int, float))
    def from_online_srtm_database(bounds, path_to_dem_file=DEFAULT_OUTPUT, product="SRTM1", margin=0,
                                  no_data_value=-32768):
        """ Import DEM tile from SRTM3 or SRTM1 online dataset

        Based on "elevation" module. Be careful that at the moment, SRTM3 product
        does not seem to work properly.
        :param bounds:
        :param path_to_dem_file:
        :param product: "SRTM1" or "SRTM3"
        :param margin: margin (in %) around DEM
        :param no_data_value: no data filling value (default = -32768)
        :return:
        """
        from subprocess import CalledProcessError
        import elevation

        try:
            check_string(product, {'SRTM1', 'SRTM3'})
        except ValueError as e:
            raise DigitalElevationModelError("Invalid product name '%s': %s" % (product, e))

        try:
            elevation.clip(bounds, output=path_to_dem_file, margin="%s" % (margin/100), product=product)
        except CalledProcessError as e:
            raise DigitalElevationModelError("Internal subprocess error: %s" % e)
        except (ValueError, TypeError, KeyError) as e:
            raise DigitalElevationModelError("Invalid input argument: %s" % e)

        # Return instance of DigitalElevationModel
        return DigitalElevationModel(path_to_dem_file, no_data_value=no_data_value)

    @staticmethod
    def from_cgiar_online_database(bounds, margin=0, max_tiles=4, no_data_value=-32768):
        """ Import DEM tile from CGIAR-CSI SRTM3 database (V4.1)

        :param bounds: bounds of the image --> (x_min, y_min, x_max, y_max)
        :param margin: margin (in %) around DEM
        :param max_tiles: max number of tiles to download
        :param no_data_value: no data filling value (default = -32768)
        :return:
        """
        srtm_lon = np.arange(-180, 185, 5)
        srtm_lat = np.arange(60, -65, -5)
        bounds = (bounds[0] - (bounds[2] - bounds[0]) * margin/100, bounds[1] - (bounds[3] - bounds[1]) * margin/100,
                  bounds[2] + (bounds[2] - bounds[0]) * margin/100, bounds[3] + (bounds[3] - bounds[1]) * margin/100)
        x_min, x_max = np.digitize(bounds[0], srtm_lon, right=True), np.digitize(bounds[2], srtm_lon, right=True)
        y_min, y_max = np.digitize(bounds[3], srtm_lat), np.digitize(bounds[1], srtm_lat)

        if (x_max - x_min) * (y_max-y_min) > max_tiles:
            raise DigitalElevationModelError("Too much tiles to download (>%d)" % max_tiles)

        list_of_tiles = []

        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                tile_temp_file = _download_srtm_tile("srtm_%02d_%02d" % (x, y))
                list_of_tiles.append(DigitalElevationModel(tile_temp_file, no_data_value=no_data_value))

        # Merge DEMs
        return DigitalElevationModel.merge(list_of_tiles, bounds)


def _download_srtm_tile(tile_name):
    """ Download and extract SRTM tile archive

    :param tile_name:
    :return:
    """
    zip_name, tif_name = tile_name + ".zip", tile_name + '.tif'
    url = os.path.join(CGIAR_URL, zip_name)
    temp_srtm_zip = os.path.join(tempfile.gettempdir(), zip_name)
    temp_srtm_dir = os.path.join(tempfile.gettempdir(), tile_name)

    # Download tile
    try:
        urlretrieve(url, temp_srtm_zip)
    except URLError as e:
        raise RuntimeError("Unable to fetch data at '%s': %s" % (url, e))

    # Extract GeoTiff
    archive = ZipFile(temp_srtm_zip, 'r')
    archive.extractall(temp_srtm_dir)
    archive.close()

    return os.path.join(temp_srtm_dir, tif_name)


if __name__ == "__main__":
    test = RasterMap("/home/benjamin/Documents/Data/Resource rasters/Solar maps/Monthly GHI sum/monthly_GHI_01.tif")
    m0 = test[10:200]
    m = test[10:200, 10:200]
    m2 = test[test > 1278]
    m3 = test[7, 8]
    print((m0 == m).all())
    print("done")
