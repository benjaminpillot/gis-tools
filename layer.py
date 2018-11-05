# -*- coding: utf-8 -*-

""" Geo layer object classes and methods

Toolset for working with static geo layer elements (networks, buffers, areas such as administrative boundaries,
road/electrical networks, waterways, restricted areas, etc.)
"""
import math
import os

import fiona
import warnings

import pyproj
from copy import copy
from functools import wraps
import geopandas as gpd
import numpy as np
import copy

from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, shape, MultiPoint
from fiona.errors import FionaValueError
from geopandas.io.file import infer_schema

from geotools.distance import compute_distance
from geotools.conversion import geopandas_to_array
from geotools.coordinates import GeoGrid, r_tree_idx
from geotools.exceptions import GeoLayerError, GeoLayerWarning, LineLayerError, PointLayerError, PolygonLayerError
from geotools.geometry import katana, fishnet, explode, cut, cut_, cut_at_points, add_points_to_line, \
    radius_of_curvature, shared_area_among_collection, intersects, intersecting_features, katana_centroid
from geotools.plotting import plot_geolayer
from geotools.projections import is_equal, proj4_from, ellipsoid_from, proj4_from_layer
from toolset.list import split_list_by_index
from utils.check import check_type, check_string, type_assert, protected_property
from utils.check.value import check_sub_collection_in_collection

# __all__ = []
# __version__ = '0.1'
from utils.sys.timer import Timer

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2017, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


# Decorator for returning new instance of GeoLayer and subclasses
def return_new_instance(method):
    @wraps(method)
    def _return_new_instance(self, *args, **kwargs):
        output = method(self, *args, **kwargs)
        if isinstance(output, gpd.GeoDataFrame):
            new_self = self.__class__(output, self.name)
            return new_self
        else:
            return output
    return _return_new_instance


class GeoLayer:
    """ GeoLayer base class

    Use this base class and its sub-classes
    in order to implement specific geo layers (from
    geo file or geopandas datasets)
    """

    # gpd_df = protected_property('gpd_df')
    geom_type = protected_property('geom_type')

    _split_methods = None
    _split_threshold = None

    _geometry_class = None
    _multi_geometry_class = None

    def __init__(self, layer_to_set, name: str='layer'):
        """ GeoLayer constructor

        :param layer_to_set: geo file or geopandas data frame
        """

        check_type(layer_to_set, (str, gpd.GeoDataFrame), name, str)

        if type(layer_to_set) == str:
            try:
                self._file = layer_to_set
                gpd_df = gpd.read_file(layer_to_set)
            except (OSError, FionaValueError) as e:
                raise GeoLayerError("Impossible to load file {}:\n{}".format(layer_to_set, e))
            except (AttributeError, ValueError):
                # Sometimes, only one geometry could be wrong... And GeoPandas read_file method will not succeed
                # See https://gis.stackexchange.com/questions/277231/geopandas-valueerror-a-linearring-must-have-at
                # -least-3-coordinate-tuples
                input_collection = list(fiona.open(layer_to_set, 'r'))
                output_collection = []
                for element in input_collection:
                    try:
                        shape(element["geometry"])
                        output_collection.append(element)
                    except (AttributeError, ValueError):
                        pass
                gpd_df = gpd.GeoDataFrame.from_features(output_collection, proj4_from_layer(layer_to_set))
        else:
            gpd_df = gpd.GeoDataFrame().append(layer_to_set, ignore_index=True)

        if len(gpd_df) == 0:
            raise GeoLayerError("Empty geo-layer dataset")

        if 'geometry' not in gpd_df.keys():
            raise GeoLayerError("Geometry is not defined in dataset")

        # Geo layer must own consistent geometry
        geom_type = {'LineString': "Line", "MultiLineString": "Line", "Polygon": "Polygon", "MultiPolygon":
                     "Polygon", "Point": "Point"}
        geometry = [geom_type[geom] for geom in gpd_df["geometry"].type]
        if len(np.unique(geometry)) > 1:
            raise GeoLayerError("Layer geometry must be consistent")

        # Set attributes
        self._geom_type = geometry[0]
        self._gpd_df = gpd_df
        self._point_layer_class = PointLayer  # Point layer class/subclass corresponding to given layer class/subclass
        self._polygon_layer_class = PolygonLayer  # Polygon layer class/subclass corresponding to given layer
        # class/subclass
        self._line_layer_class = LineLayer  # Line layer class/subclass corresponding to given layer class/subclass
        self.name = name

    @return_new_instance
    def add_points_to_geometry(self, distance):
        """

        :param distance:
        :return:
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = [add_points_to_line(geom, distance) for geom in self.exterior]

        return outdf

    @return_new_instance
    def add_z(self, dem, no_data_value=0):
        """ Add z dimension to layer

        :param dem: DigitalElevationModel instance
        :param no_data_value: value by which NaN must be replaced
        :return:
        """
        from geotools.raster import DigitalElevationModel

        check_type(dem, DigitalElevationModel)
        self.check_proj(self.crs, dem.crs)

        geometry = []
        for i, x, y in self.iterxy():
            z = np.array([dem.get_value_at(yi, xi) for xi, yi in zip(x, y)])
            z[np.isnan(z)] = no_data_value
            geometry.append(self.geometry[i].__class__([(xi, yi, zi) for xi, yi, zi in zip(x, y, z)]))

        outdf = self._gpd_df.copy()
        outdf.geometry = geometry

        return outdf

    @return_new_instance
    def append(self, other):
        """ Append other GeoLayer to instance

        :param other: GeoLayer instance of the same class as self
        :return:
        """
        if self.__class__ != other.__class__:
            raise GeoLayerError("Must append '%s' but input is '%s'" % (type(self).__name__, type(other).__name__))

        other = other.to_crs(self.crs)

        return self._gpd_df.append(other._gpd_df, ignore_index=True, sort=False)

    @type_assert(distance=(int, float))
    def buffer(self, distance, resolution=16):
        """ Return buffer geometry

        Return buffer zone(s) around object
        :param distance: radius of the buffer zone
        :param resolution:
        :return: PolygonLayer instance
        """
        buffer = self._gpd_df.buffer(distance, resolution)
        outdf = self._gpd_df.copy()
        outdf.geometry = buffer

        return self._polygon_layer_class(outdf, self.name)

    def centroid(self):
        """

        :return: PointLayer instance
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = self._gpd_df.centroid

        return self._point_layer_class(outdf, self.name)

    @classmethod
    def check_proj(cls, *args, warning=True):
        """ Check equality of projections

        :param args: input CRS
        :param warning: if True, only raise a warning
        :return:
        """
        if len(args) < 2:
            raise ValueError("At least 2 inputs are required")

        for crs1, crs2 in zip(args[:-1], args[1::]):
            if not is_equal(crs1, crs2) and not warning:
                raise GeoLayerError("Projections should be the same but are '%s' and '%s'" % (crs1, crs2))
            elif not is_equal(crs1, crs2) and warning:
                print("Different projections ('%s' and '%s') might give unexpected results" % (crs1, crs2))

    @return_new_instance
    def convex_hull(self):
        """ Return convex hull

        :return:
        """
        return self._gpd_df.convex_hull

    def distance(self, other):
        """ Get min distance to other layer

        Compute min distance to other layer,
        for each feature (no element-wise).
        Return a numpy array
        :param other: GeoLayer instance
        :return:
        """

        return self._distance_and_nearest_neighbor(other, False)[0]

    def distance_and_nearest_neighbor(self, other):
        """ Get both min distance and nearest neighbor

        :param other:
        :return:
        """

        return self._distance_and_nearest_neighbor(other, True)

    @return_new_instance
    def drop(self, labels):
        """ Drop rows according to labels

        :param labels:
        :return:
        """
        return self._gpd_df.drop(labels)

    @return_new_instance
    def explode(self):
        """ Explode "multi" geometry into "single"

        Thanks to https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26
        :return:
        """
        outdf = gpd.GeoDataFrame(columns=self._gpd_df.columns, crs=self.crs)
        append_bool = np.full(len(self), False)
        for idx, row in self.iterrows():
            if type(row.geometry) == self._geometry_class:
                append_bool[idx] = True
            if type(row.geometry) == self._multi_geometry_class:
                outdf = outdf.append(self._gpd_df[append_bool], ignore_index=True)
                append_bool[append_bool] = False
                multdf = gpd.GeoDataFrame(columns=self._gpd_df.columns)
                multdf = multdf.append([row]*len(row.geometry), ignore_index=True)
                multdf.geometry = list(row.geometry)
                outdf = outdf.append(multdf, ignore_index=True)

        return outdf.append(self._gpd_df[append_bool], ignore_index=True)

    def get_underlying_points_as_new_layer(self, location=None):
        """ Get underlying points constituting the layer as a new PointLayer instance

        Get underlying point coordinates of the layer as a new point layer.
        If location is None, all points are converted
        :param location: list of tuples with 2 elements (first: object number, second: point index within coordinate
        list)
        :return: PointLayer class instance
        """
        outdf = gpd.GeoDataFrame(columns=self._gpd_df.columns, crs=self.crs)
        new_geom = []

        if location is None:
            for n in range(len(self)):
                coords = list(self.exterior[n].coords)
                new_geom.extend([Point(c) for c in coords])
                outdf = outdf.append([self._gpd_df.iloc[n]] * len(coords), ignore_index=True)
        else:
            for loc in location:
                coords = list(self.exterior[loc[0]].coords)
                new_geom.append(Point(coords[loc[1]]))
                outdf = outdf.append(self._gpd_df.iloc[loc[0]], ignore_index=True)

        outdf.geometry = new_geom

        return self._point_layer_class(outdf, self.name)

    def intersects(self, other):
        """

        Retrieve which elements of other intersect
        with elements of geo layer (not element-wise)
        :param other:
        :return:
        """
        check_type(other, GeoLayer)
        list_of_intersecting_features = []
        idx = r_tree_idx(other.geometry)

        for geom in self.geometry:
            list_of_intersecting_features.append(intersecting_features(geom, other.geometry, idx))

            # list_of_intersecting_features = list(idx.intersection(geom.bounds))
            # intersects = [geom.intersects(other.geometry[f]) for f in list_of_intersecting_features]
            # intersecting_features.append([f for i, f in enumerate(list_of_intersecting_features) if intersects[i]])

        return list_of_intersecting_features

    def is_exploded(self):
        """ Test if geometry is "exploded"

        Test whether geometry is exploded, that is
        whether geometry is only single-part and not
        multi-part
        :return:
        """
        exploded_geom = explode(self.geometry)
        if len(self) == len(exploded_geom):
            return True
        else:
            return False

    def is_valid(self):
        """ Is geometry valid ?

        :return:
        """
        return self._gpd_df.is_valid

    def iterrows(self):
        return self._gpd_df.iterrows()

    def iterxy(self):
        """ Iterate over x and y coordinates of
        all layer's geometries

        :return: tuple of arrays
        """
        num = 0
        while num < len(self):
            yield self.index[num], self.xy(num)[0], self.xy(num)[1]
            num += 1

    def iterxy_in_geometry(self, n):
        """ Iterate over x and y coords of given geometry

        :param n: geometry's index
        :return:
        """
        num = 0
        while num < len(self.xy(n)[0]):
            yield self.xy(n)[0][num], self.xy(n)[1][num]
            num += 1

    def length_xy_of_geometry(self, n):
        """ Compute 2D length of given geometry

        :param n:
        :return: numpy array of length values
        """
        x, y = np.array(self.xy(n)[0]), np.array(self.xy(n)[1])
        return np.sqrt((x[1::] - x[:-1:]) ** 2 + (y[1::] - y[:-1:]) ** 2)

    def length_xyz_of_geometry(self, n):
        """ Compute 3D length of given geometry

        :param n:
        :return: numpy array of length values
        """
        x, y = np.array(self.xy(n)[0]), np.array(self.xy(n)[1])
        if self.exterior[n].has_z:
            z = np.array(self.exterior[n].coords)[:, 2]
        else:
            z = np.zeros(len(x))

        return np.sqrt((x[1::] - x[:-1:]) ** 2 + (y[1::] - y[:-1:]) ** 2 + (z[1::] - z[:-1:]) ** 2)

    def length_xyz(self):
        """ Compute length with Z dimension

        :return:
        """
        length = []
        for geom in self.exterior:
            if geom.has_z:
                x, y = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
                z = np.array(geom.coords)[:, 2]
                length.append(np.sum(np.sqrt((x[1::] - x[:-1:])**2 + (y[1::] - y[:-1:])**2 + (z[1::] - z[:-1:])**2)))
            else:
                length.append(geom.length)

        return length

    @type_assert(geo_grid=GeoGrid)
    def min_distance_to_layer(self, geo_grid: GeoGrid):
        """ Compute minimum distance to layer

        Compute minimum distance between points
        from a defined georeferenced grid and the
        point coordinates of the layer.
        Return an array of min distance values

        :param geo_grid: GeoGrid class instance
        :return:

        :Example:
            *
        """
        # Initialize min distance array
        min_distance = np.full(geo_grid.latitude_mesh.shape, 1e9)

        # Iterate over all geometries of layer objects
        for obj in self.geometry:
            for x, y in zip(obj.coords.xy[0], obj.coords.xy[1]):
                distance = compute_distance(x, y, geo_grid.longitude_mesh, geo_grid.latitude_mesh, geo_grid.type,
                                            ellipsoid_from(self.crs))
                min_distance = np.minimum(distance, min_distance)

        return min_distance

    def nearest_neighbor(self, other):
        """ Get nearest neighbor in other layer

        :param other:
        :return:
        """

        return self._distance_and_nearest_neighbor(other, True)[1]

    def nearest_neighbors(self, other, buffer_distance):
        """ Get nearest neighbors of other layer

        Get nearest neighbor elements of other layer
        within a specific buffer around each element
        of geo layer
        :param other:
        :param buffer_distance:
        :return: numpy array with shape = (length(self), length(other)) such as
        cell(n, m) stands for distance between element n of layer and element m
        of other. If element m is not a nearest neighbor, NaN is returned
        """
        buffered_layer = self.buffer(buffer_distance)

        self.check_proj(self.crs, other.crs)

        # Use Rtree to speed up !
        idx = r_tree_idx(other.geometry)

        nearest_neighbors = np.full((len(self), len(other)), np.nan)

        for i, buffer_geom in enumerate(buffered_layer.geometry):
            list_of_intersecting_features = list(idx.intersection(buffer_geom.bounds))
            buffer_intersects = [buffer_geom.intersects(other.geometry[f]) for f in list_of_intersecting_features]
            if any(buffer_intersects):
                list_of_buffer_intersecting_features = [f for i, f in enumerate(list_of_intersecting_features) if
                                                        buffer_intersects[i]]
                distance = []
                for feature in list_of_buffer_intersecting_features:
                    distance.append(self.geometry[i].distance(other.geometry[feature]))
                nearest_neighbors[i, [f for f in list_of_buffer_intersecting_features]] = [d for d in distance]
                # intersects = [self.geometry[i].intersects(other.geometry[f]) for f in list_of_intersecting_features]
                # if any(intersects):
                #     list_of_truly_intersecting_features = [f for i, f in enumerate(list_of_intersecting_features) if
                #                                            intersects[i]]
                #     nearest_neighbors[i, [f for f in list_of_truly_intersecting_features]] = 0

        return nearest_neighbors

    def nearest_point_in_layer(self, points):
        """ Find nearest points in layer elements

        Find nearest point in layer geometry coordinates.
        Return element row index and index of the point within
        coordinate sequence for each element (row) of the data frame
        :param points: PointLayer
        :return:
        """

        check_type(points, PointLayer)

        # Get nearest element index in other layer
        nearest_neighbor = points.nearest_neighbor(self)

        # Output
        result = []

        for n, geom in enumerate(points.geometry):
            x, y = np.array(self.exterior[nearest_neighbor[n]].coords.xy[0]), np.array(self.exterior[nearest_neighbor[
                n]].coords.xy[1])
            argmin = np.argmin(np.sqrt((geom.x - x) ** 2 + (geom.y - y) ** 2))
            result.append((nearest_neighbor[n], argmin))

        return sorted(result)

    def project(self, other):
        """ Project layer geometry onto other geometry

        Projection results in PointLayer instance with
        points belonging to other geometry and nearest
        to layer objects
        :param other:
        :return: PointLayer with points belonging to other geometry
        """
        nn = self.nearest_neighbor(other)
        outdf = self._gpd_df.copy()

        for idx, n in enumerate(nn):
            list_of_points = MultiPoint(self.exterior[idx].coords)
            distance = [other.geometry[n].distance(point) for point in list_of_points]
            point = list_of_points[np.argmin(distance)]
            d = other.geometry[n].project(point)
            p = other.geometry[n].interpolate(d)
            outdf.loc[idx, 'geometry'] = p

        return self._point_layer_class(outdf, self.name)

    def simplify(self, tolerance=0.2):
        """ Simplify object geometry

        :param tolerance:
        :return: GeoLayer
        """
        out_layer = self.copy()
        out_layer['geometry'] = self.geometry.simplify(tolerance)
        return out_layer

    @return_new_instance
    def split(self, threshold, method=None, no_multipart=None):
        """ Split geometry

        :param threshold: surface threshold
        :param method: method used to split geometry
        :param no_multipart: (bool) should resulting geometry be single-part (no multi-part) ?
        :return:
        """

        if self._geom_type == 'Point':
            return self.copy()

        method = check_string(method, self._split_methods.keys())

        outdf = gpd.GeoDataFrame(columns=self._gpd_df.columns, crs=self.crs)
        append_bool = np.full(len(self), False)
        for idx, row in self.iterrows():
            if row.geometry.__getattribute__(self._split_threshold) <= threshold:
                append_bool[idx] = True
            else:
                outdf = outdf.append(self._gpd_df[append_bool], ignore_index=True)
                append_bool[append_bool] = False
                multdf = gpd.GeoDataFrame(columns=self._gpd_df.columns)
                split_geom = self._split_methods[method](row.geometry, threshold)
                if no_multipart:
                    split_geom = explode(split_geom)
                multdf = multdf.append([row]*len(split_geom), ignore_index=True)
                multdf.geometry = split_geom
                outdf = outdf.append(multdf, ignore_index=True)
        return outdf

    def to_array(self, geo_grid: GeoGrid, attribute, data_type='uint8', all_touched=False):
        """ Convert layer to numpy array

        :param geo_grid: GeoGrid instance
        :param attribute: valid attribute of GeoLayer dataset
        :param data_type:
        :param all_touched: boolean --> rasterization type (cell centers or "all touched")
        :return: numpy array
        """
        check_type(geo_grid, GeoGrid, attribute, str)

        if attribute not in self.attributes():
            raise GeoLayerError("%s is not a valid attribute" % attribute)

        return geopandas_to_array(self._gpd_df, attribute, geo_grid, data_type, all_touched)

    def to_raster_map(self, geo_grid: GeoGrid, attribute):
        """ Convert layer to RasterMap instance

        :param geo_grid:
        :param attribute:
        :return:
        """
        from geotools.raster import RasterMap
        return RasterMap(self.to_array(geo_grid, attribute), geo_grid)

    @return_new_instance
    def to_crs(self, crs=None, epsg=None):
        if crs is not None:
            try:
                crs = proj4_from(crs)
            except ValueError:
                raise GeoLayerError("Unable to convert given CRS to proj4 format")

        if epsg is not None and crs is None:
            try:
                crs = proj4_from(epsg)
            except ValueError:
                raise GeoLayerError("Invalid EPSG code")
        elif epsg is not None and crs is not None:
            warnings.warn("CRS already given. Skip EPSG code...", GeoLayerWarning)

        if crs is not None:
            if not is_equal(crs, self.crs):
                return self._gpd_df.to_crs(crs=crs)
            else:
                return self.copy()
        else:
            raise GeoLayerError("Must set either crs or epsg code")

    @type_assert(file_path=str)
    def to_csv(self, file_path, attributes=None, *args, **kwargs):
        """ Write layer to csv file

        :param file_path: path to csv file
        :param attributes: layer attributes to write in csv
        :param args:
        :param kwargs:
        :return:
        """
        if attributes is not None:
            check_sub_collection_in_collection(attributes, self.attributes())

        self._gpd_df.to_csv(file_path, columns=attributes, *args, **kwargs)

    def to_file(self, file_path, driver="ESRI Shapefile", **kwargs):
        """ Write geo layer to file

        :param file_path:
        :param driver: valid fiona driver (fiona.supported_drivers)
        :return:
        """
        import fiona
        driver = check_string(driver, list(fiona.supported_drivers.keys()))
        try:
            os.remove(file_path)
        except OSError:
            pass
        self._gpd_df.to_file(file_path, driver, **kwargs)

    def xy(self, n):
        """ Return xy coords of geo layer nth geometry

        :param n: geometry's index
        :return:
        """
        return self.exterior[n].coords.xy[0], self.exterior[n].coords.xy[1]

    def attributes(self):

        return self._gpd_df.keys()

    def plot(self, *args, **kwargs):

        return plot_geolayer(self, *args, **kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def shallow_copy(self):

        return copy.copy(self)

    def __len__(self):

        return len(self._gpd_df)

    def __repr__(self):
        return repr(self._gpd_df)

    # __getitem__ method returns new instance or pandas Series
    @return_new_instance
    def __getitem__(self, key):
        try:
            return self._gpd_df[key].copy()  # .copy()
        except KeyError:
            raise GeoLayerError("'%s' is not a valid attribute" % key)

    def __setitem__(self, key, value):

        self._gpd_df[key] = value

        return self

    @property
    def bounds(self):
        return self._gpd_df.total_bounds

    @property
    def crs(self):
        return self.pyproj.srs

    @property
    def exterior(self):
        return self.geometry

    @property
    def geometry(self):
        return self._gpd_df.geometry

    @property
    def geo_type(self):
        if self.pyproj.is_latlong():
            return "latlon"
        else:
            return "equal"

    @property
    def index(self):
        return self._gpd_df.index

    @property
    def length(self):
        return self._gpd_df.geometry.length

    @property
    def pyproj(self):
        return pyproj.Proj(self._gpd_df.crs)

    @property
    def schema(self):
        return infer_schema(self._gpd_df)

    ###################
    # Protected methods

    def _distance_and_nearest_neighbor(self, other, compute_nearest_neighbor):
        """ Get min distance and NN of other layer

        :param other: GeoLayer instance
        :param compute_nearest_neighbor: boolean
        :return:
        """
        check_type(other, GeoLayer)
        self.check_proj(self.crs, other.crs)

        # Use Rtree to speed up !
        idx = r_tree_idx(other.geometry)

        min_dist = np.zeros(len(self))
        nearest_neighbor = np.zeros(len(self), dtype='int')

        for i, geom in enumerate(self.geometry):
            list_of_nearest_features = list(idx.nearest(geom.bounds, 1))
            list_of_intersecting_features = list(idx.intersection(geom.bounds))
            is_intersecting = [geom.intersects(other.geometry[f]) for f in list_of_intersecting_features]
            dist = []
            if not any(is_intersecting):
                for f in list_of_nearest_features:
                    dist.append(geom.distance(other.geometry[f]))
                min_dist[i] = np.min(dist)
                if compute_nearest_neighbor:
                    nearest_feature = int(np.argmin(dist) % len(list_of_nearest_features))
                    nearest_neighbor[i] = list_of_nearest_features[nearest_feature]
            else:
                if compute_nearest_neighbor:
                    list_of_truly_intersecting_features = [f for i, f in enumerate(list_of_intersecting_features) if
                                                           is_intersecting[i]]
                    for f in list_of_truly_intersecting_features:
                        dist.append(geom.centroid.distance(other.geometry[f].centroid))
                    nearest_neighbor[i] = list_of_truly_intersecting_features[int(np.argmin(dist) % len(
                        list_of_truly_intersecting_features))]

        return min_dist, nearest_neighbor


class PolygonLayer(GeoLayer):
    """ Polygon layer instance

    Geo layer of polygons
    """

    _split_methods = {'katana_simple': katana, 'katana_centroid': katana_centroid, 'fishnet': fishnet}
    _split_threshold = 'area'
    _geometry_class = Polygon
    _multi_geometry_class = MultiPolygon

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self._geom_type != 'Polygon':
            raise PolygonLayerError("Geometry must be 'Polygon' but is '{}'".format(self._geom_type))

    def attr_area(self, other, attr_name: str, normalized: bool=False):
        """ Area of attribute from other PolygonLayer in current layer

        Spatial join which gives area of specific attribute value
        in other PolygonLayer intersecting current layer
        :param other: PolygonLayer class instance
        :param attr_name: Attribute name in other PolygonLayer
        :param normalized: is area normalized with respect to each layer's polygon area ?
        :return:
        """
        check_type(other, PolygonLayer, attr_name, str, normalized, bool)
        self.check_proj(self.crs, other.crs)

        idx = r_tree_idx(other.geometry)
        attr_value = {attr: np.zeros(len(self)) for attr in set(other[attr_name])}
        for i, geom in enumerate(self.geometry):
            area = np.array(shared_area_among_collection(geom, other.geometry, normalized, idx))
            for val, _area in zip(other[attr_name][area > 0], area[area > 0]):
                attr_value[val][i] += _area

        return attr_value

    def clean_geometry(self, delete_invalid=False):
        """ Clean invalid geometries

        :return:
        """
        layer = self.buffer(0, 0)

        if delete_invalid:
            return layer[layer.is_valid()]
        else:
            return layer

    @return_new_instance
    def overlay(self, other, how):
        """ Apply overlay geometry operation from another PolygonLayer

        :param other:
        :param how:
        :return:
        """
        from pandas import concat
        check_type(other, PolygonLayer)
        how = check_string(how, ("intersection", "difference", "union", "symmetric_difference", "identity"))

        idx = r_tree_idx(other.geometry)
        new_geom = []
        if how == "intersection":
            outdf = gpd.GeoDataFrame(columns=list(self.attributes()) + list(other.attributes()), crs=self.crs)
            for i, row in self.iterrows():
                is_intersecting = intersects(row.geometry, other.geometry, idx)
                new_geom.extend([row.geometry.intersection(geom) for geom in other.geometry[is_intersecting]])
                df = gpd.GeoDataFrame(columns=self._gpd_df.columns)
                if any(is_intersecting):
                    other_layer = other[is_intersecting]
                    df = df.append([row] * len(other_layer), ignore_index=True)
                    outdf = outdf.append(concat([df, other_layer._gpd_df], axis=1), ignore_index=True)

        elif how == "difference":
            outdf = gpd.GeoDataFrame(columns=self.attributes(), crs=self.crs)
            for i, row in self.iterrows():
                is_intersecting = intersects(row.geometry, other.geometry, idx)
                if any(is_intersecting):
                    diff_result = explode([row.geometry.difference(geom) for geom in other.geometry[is_intersecting]])
                    new_geom.extend(diff_result)
                    if len(diff_result) > 0:
                        outdf = outdf.append([row] * len(diff_result), ignore_index=True)
                else:
                    new_geom.append(row.geometry)
                    outdf = outdf.append(row, ignore_index=True)

        else:
            outdf = self._gpd_df.copy()

        if len(outdf) == 0:
            raise PolygonLayerError("Resulting layer is empty")

        outdf = outdf.drop(columns=["geometry"])
        outdf.geometry = new_geom

        return outdf

    def split(self, threshold, method="katana_simple", no_multipart=False):
        """

        :param threshold:
        :param method:
        :param no_multipart:
        :return:
        """
        return super().split(threshold, method, no_multipart)

    @property
    def area(self):
        return self.geometry.area

    @property
    def exterior(self):
        return self.geometry.exterior


class LineLayer(GeoLayer):
    """ Line layer instance

    Geo layer with only line geometry
    """

    _split_methods = {'cut': cut, 'cut_': cut_}
    _split_threshold = 'length'
    _geometry_class = LineString
    _multi_geometry_class = MultiLineString

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Check geometry
        if self._geom_type != 'Line':
            raise LineLayerError("Geometry of LineLayer must be 'Line' but is '{}'".format(self._geom_type))

    def radius_of_curvature(self, n):
        """ Compute road's radius of curvature

        :param n:
        :return:
        """
        return radius_of_curvature(self.geometry[n])

    def slope(self, slope_format="percent"):
        """ Compute 3D line slope

        : param slope_format:
        :return: list of slope values
        """
        slope_format = check_string(slope_format, {'degree', 'percent'})
        slope = []
        for geom in self.geometry:
            if geom.has_z:
                if slope_format == "percent":
                    slope.append(100 * (geom.coords[-1][2] - geom.coords[0][2])/geom.length)
                else:
                    slope.append(math.atan((geom.coords[-1][2] - geom.coords[0][2])/geom.length) * 180/math.pi)
            else:
                slope.append(0)

        return slope

    def slope_of_geometry(self, n, slope_format="percent", z_spatial_resolution=0):
        """ Compute 3D slope of given geometry

        :param n: geometry index
        :param slope_format:
        :param z_spatial_resolution: spatial accuracy on Z estimates (e.g.: DEM resolution from which Z has been
        derived)
        :return:
        """
        slope_format = check_string(slope_format, {'degree', 'percent'})
        if self.geometry[n].has_z:
            z = np.array(self.exterior[n].coords)[:, 2]
            if slope_format == "percent":
                slope = 100 * (z[1::] - z[:-1:])/np.maximum(z_spatial_resolution, self.length_xy_of_geometry(n))
            else:
                slope = np.arctan((z[1::] - z[:-1:])/np.maximum(z_spatial_resolution, self.length_xy_of_geometry(n)))\
                        * 180/np.pi
        else:
            slope = np.zeros(len(self.exterior[n].coords))

        return slope

    def split(self, threshold, method="cut", no_multipart=False):
        """

        :param threshold:
        :param method:
        :param no_multipart:
        :return:
        """
        return super().split(threshold, method, no_multipart)

    def split_at_intersections(self):
        # TODO: split lines with lines. Cut lines at any intersection point of the layer
        pass

    @return_new_instance
    def split_at_points(self, points, tolerance=1):
        """ Split lines at given points

        :param points: PointLayer instance
        :param tolerance: maximum distance threshold for
        considering point near enough from line (default: 1m)
        :return:
        """
        check_type(points, PointLayer)
        distance, nearest_neighbor = points.distance_and_nearest_neighbor(self)
        out_df = self._gpd_df.copy()

        multdf = gpd.GeoDataFrame(columns=self._gpd_df.columns)
        geometry = []

        for nn in nearest_neighbor:
            cut_points = points[nearest_neighbor == nn]
            new_geom = cut_at_points(out_df.geometry[nn], cut_points.geometry)
            multdf = multdf.append([out_df.iloc[nn]] * len(new_geom), ignore_index=True)
            geometry.extend(new_geom)

        multdf.geometry = geometry
        out_df = out_df.append(multdf, ignore_index=True)

        return out_df

    @return_new_instance
    def split_at_underlying_points(self, location):
        """ Split layer elements at existing coordinates

        :param location: list of tuples with 2 elements (object number, index within coordinate sequence)
        :return:
        """
        outdf = gpd.GeoDataFrame(columns=self._gpd_df.columns, crs=self.crs)
        new_geom = []
        for n in range(len(self)):
            coords = list(self.geometry[n].coords)
            break_idx = [loc[1] for loc in location if loc[0] == n and 0 < loc[1] < len(coords) - 1]
            if len(break_idx) == 0:
                outdf = outdf.append(self._gpd_df.iloc[n], ignore_index=True)
                new_geom.append(self.geometry[n])
            else:
                new_coords = split_list_by_index(coords, break_idx, include=True)
                outdf = outdf.append([self._gpd_df.iloc[n]] * len(new_coords), ignore_index=True)
                new_geom.extend([LineString(c) for c in new_coords])

        outdf.geometry = new_geom

        return outdf


class PointLayer(GeoLayer):
    """ Point layer instance

    Geo layer with only point geometry
    """
    _geometry_class = Point
    _multi_geometry_class = MultiPoint

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self._geom_type != 'Point':
            raise PointLayerError("Geometry must be 'Point' but is '{}'".format(self._geom_type))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    solar = PolygonLayer("/home/benjamin/Documents/Data/Results/Solar/result.geojson")
    landuse = PolygonLayer("/home/benjamin/Documents/Data/Geo layers/landuse/zone_occupation_sol.shp")

    try:
        test = solar.overlay(solar, "difference")
    except PolygonLayerError:
        print("this is empty")
    else:
        # landuse.plot()
        plt.figure(1)
        test.plot()
        plt.figure(2)
        solar.plot()
        plt.show()
