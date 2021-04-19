# -*- coding: utf-8 -*-

""" Geo layer object classes and methods

Toolset for working with static geo layer elements (networks, buffers,
areas such as administrative boundaries, road/electrical networks,
 waterways, restricted areas, etc.)
"""
import copy
import math
import os
import random
import warnings
from functools import wraps

import fiona
import geopandas as gpd
import numpy as np
# import progressbar
import pyproj
from fiona.errors import FionaValueError
from geopandas.io.file import infer_schema
from gistools.conversion import geopandas_to_array
from gistools.coordinates import GeoGrid, r_tree_idx
from gistools.exceptions import GeoLayerError, LineLayerError, PointLayerError, \
    PolygonLayerError, PolygonLayerWarning, GeoLayerEmptyError, ProjectionWarning
from gistools.geometry import katana, fishnet, explode, cut, cut_, cut_at_points, \
    add_points_to_line, radius_of_curvature, shared_area_among_collection, intersects, \
    intersecting_features, katana_centroid, area_partition_polygon, shape_factor, \
    is_in_collection, overlapping_features, overlaps, hexana, nearest_feature, to_2d
from gistools.plotting import plot_geolayer
from gistools.projections import crs_from_layer
from numba import njit
from pandas import concat, Series
from rdp import rdp
from shapely import wkb
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, \
    Point, shape, MultiPoint
from shapely.ops import cascaded_union
from shapely.prepared import prep

from gistools.utils.check.descriptor import protected_property, lazyproperty
from gistools.utils.check.type import check_type, type_assert
from gistools.utils.check.value import check_string, check_sub_collection_in_collection
from gistools.utils.toolset import split_list_by_index
from tqdm import tqdm


def _build_consistent_gdf(data, layer_class, **kwargs):
    """ Build geopandas dataframe with consistent geometry

    Eliminate inconsistent geometries (keep only consistent one,
    e.g. only lines, points or polygons)

    Parameters
    ----------
    data:
    layer_class:
    kwargs:

    Returns
    -------
    """
    outdf = gpd.GeoDataFrame(data, **kwargs)
    outdf = outdf[outdf.geometry.apply(
        lambda geom: isinstance(geom, (layer_class._geometry_class,
                                       layer_class._multi_geometry_class)))]

    return outdf


def _difference(layer1, layer2):
    """ Difference between two layers

    Parameters
    ----------
    layer1
    layer2

    Returns
    -------

    """
    gdf1 = layer1._gpd_df.drop("geometry", axis=1)
    new_geometry = []
    df = []
    for i, geometry in enumerate(layer1.geometry):
        is_intersecting = intersects(geometry, layer2.geometry, layer2.r_tree_idx)
        if any(is_intersecting):
            diff_result = explode([geometry.difference(cascaded_union(
                [geom for geom in layer2.geometry[is_intersecting]]))])
            new_geometry.extend(diff_result)
            if len(diff_result) > 0:
                df.extend([gdf1.iloc[[i]]] * len(diff_result))
        else:
            new_geometry.append(geometry)
            df.append(gdf1.iloc[[i]])

    return _build_consistent_gdf(concat(df, ignore_index=True), layer1.__class__,
                                 geometry=new_geometry, crs=layer1.crs)


def _intersection(layer1, layer2):
    """ Intersection between two layers

    Parameters
    ----------
    layer1
    layer2

    Returns
    -------

    """
    new_geometry = []
    gdf1 = layer1._gpd_df.drop("geometry", axis=1)
    gdf2 = layer2._gpd_df.drop("geometry", axis=1)
    df1 = []
    df2 = []
    for i, geometry in enumerate(layer1.geometry):
        is_intersecting = intersects(geometry, layer2.geometry, layer2.r_tree_idx)
        if any(is_intersecting):
            new_geometry.extend([geometry.intersection(geom)
                                 for geom in layer2.geometry[is_intersecting]])
            df1.extend([gdf1.iloc[[i]]] * is_intersecting.count(True))
            df2.append(gdf2[is_intersecting])

    # Use the pandas concat method to speed up dataframe appending
    df1_df2 = concat([concat(df1, ignore_index=True), concat(df2, ignore_index=True)], axis=1)

    return _build_consistent_gdf(df1_df2, layer1.__class__, geometry=new_geometry, crs=layer1.crs)


def cascaded_intersection(list_of_layers):
    """ Return intersection of multiple layers

    First layer in list shall give the dimension of the result

    Parameters
    ----------
    list_of_layers: list[gistools.layer.GeoLayer]

    Returns
    -------
    """
    if len(list_of_layers) <= 1:
        return list_of_layers[0].copy()

    intersection = list_of_layers[0]
    level = 1

    while "there is a layer to intersect":
        intersection = intersection.overlay(list_of_layers[level], how='intersection')

        if level < len(list_of_layers) - 1:
            level += 1
        else:
            return intersection


def check_proj(*crs, warning=True):
    """ Check equality of projections

    Parameters
    ----------
    crs:
    warning: bool
        if True raise warning when projections are different
    """
    if len(crs) < 2:
        raise ValueError("At least 2 inputs are required")

    for crs1, crs2 in zip(crs[:-1], crs[1::]):
        if crs1 != crs2 and not warning:
            raise TypeError("Projections should be the same but are '%s' and '%s'" % (crs1, crs2))
        elif crs1 != crs2 and warning:
            warnings.warn("Different projections ('%s' and '%s') "
                          "might give unexpected results" % (crs1, crs2),
                          ProjectionWarning)


def concat_layers(list_of_layers):
    """ Concatenate layers

    Parameters
    ----------
    list_of_layers:

    Returns
    -------
    """
    df = concat([layer._gpd_df for layer in list_of_layers], sort=False, ignore_index=True)

    return list_of_layers[0].__class__.from_gpd(df, crs=list_of_layers[0].crs)


def iterate_over_geometry(replace_by_single=False):
    """ Decorator for wrapping iteration methods over geometries of layer

    Parameters
    ----------
    replace_by_single: bool
        if True, output layer has the same length,
        each geometry is replaced by a new single one. If False, output layer
        gets a new length where each geometry is replaced by multiple geometries.

    Returns
    -------
    """
    def decorate(method):

        @wraps(method)
        @return_new_instance
        def wrapper(self, *args, **kwargs):

            # Display progress bar in console if necessary
            try:
                show_progressbar = kwargs["show_progressbar"]
            except KeyError:
                show_progressbar = False

            if show_progressbar:
                geometries = tqdm(self.geometry, desc=method.__name__.lstrip('_'))
                # widgets = [method.__name__.lstrip('_'), ': ',  progressbar.Percentage(), ' ',
                #            progressbar.Bar(marker='#'), ' ', progressbar.ETA()]
                # bar = progressbar.ProgressBar(widgets=widgets, max_value=len(self)).start()
            else:
                geometries = self.geometry
                bar = None

            new_geometry = []

            # TODO: use pandas.series.apply function !!
            # Begin iteration over geometries
            if replace_by_single:
                for idx, geometry in enumerate(geometries):
                    new_geometry.append(method(self, geometry, *args, **kwargs))  # Call method here

                    # if show_progressbar:
                    #     bar.update(idx)

                # if show_progressbar:
                #     bar.finish()

                return gpd.GeoDataFrame(self._gpd_df.copy(), geometry=new_geometry, crs=self.crs)

            else:
                df = []
                try:
                    gdf = self._gpd_df.drop(self.geometry.name, axis=1)
                except KeyError:
                    gdf = self._gpd_df

                # TODO: use concat method rather than append to speed up
                for idx, geometry in enumerate(geometries):
                    new_geom = method(self, geometry, *args, **kwargs)  # Call method here
                    if new_geom:
                        df.extend([gdf.iloc[[idx]]] * len(new_geom))
                        new_geometry.extend(new_geom)
                    else:
                        df.append(gdf.iloc[[idx]])
                        new_geometry.append(geometry)

                    # if show_progressbar:
                    #     bar.update(idx)

                # if show_progressbar:
                #     bar.finish()

                return gpd.GeoDataFrame(concat(df, ignore_index=True),
                                        geometry=new_geometry, crs=self.crs)

        return wrapper
    return decorate


def return_new_instance(method):
    """ Decorator for returning new instance of GeoLayer and subclasses

    """
    @wraps(method)
    def _return_new_instance(self, *args, **kwargs):
        output = method(self, *args, **kwargs)
        if isinstance(output, gpd.GeoDataFrame):
            new_self = self.__class__(output, name=self.name)
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

    geom_type = protected_property('geom_type')

    _split_methods = None
    _split_threshold = None

    _geometry_class = None
    _multi_geometry_class = None

    _osm_type = None

    def __init__(self, layer, name: str = 'layer'):
        """ GeoLayer constructor

        Parameters
        ----------
        layer: str or geopandas.GeoDataFrame
            geo file (geojson/shape) or geopandas data frame
        name: str
            name of layer
        """

        try:
            check_type(layer, (str, gpd.GeoDataFrame), name, str)
        except TypeError as e:
            raise GeoLayerError("%s" % e)

        if type(layer) == str:
            try:
                self._file = layer
                gpd_df = gpd.read_file(layer)
            except (OSError, FionaValueError) as e:
                raise GeoLayerError("Impossible to load file {}:\n{}".format(layer, e))
            except (AttributeError, ValueError):
                # Sometimes, only one geometry could be wrong...
                # And GeoPandas read_file method will not succeed
                # See https://gis.stackexchange.com/questions/277231/geopandas-valueerror-a
                # -linearring-must-have-at-least-3-coordinate-tuples
                input_collection = list(fiona.open(layer, 'r'))
                output_collection = []
                for element in input_collection:
                    try:
                        shape(element["geometry"])
                        output_collection.append(element)
                    except (AttributeError, ValueError):
                        pass
                gpd_df = gpd.GeoDataFrame.from_features(output_collection, crs_from_layer(layer))
        else:
            gpd_df = gpd.GeoDataFrame().append(layer, ignore_index=True)

        if len(gpd_df) == 0:
            raise GeoLayerEmptyError("Empty geo-layer dataset")

        # Warning: preferable using geometry attribute as
        # it represents the active geometry for geopandas
        if not hasattr(gpd_df, "geometry"):
            raise GeoLayerError("Geometry is not defined in dataset")

        # Geo layer must own consistent geometry
        geom_type = {'LineString': "Line",
                     "MultiLineString": "Line",
                     "Polygon": "Polygon",
                     "MultiPolygon": "Polygon",
                     "Point": "Point",
                     "MultiPoint": "Point"}
        geometry = [geom_type[geom] for geom in gpd_df.geometry.type]
        if len(np.unique(geometry)) > 1:
            raise GeoLayerError("Layer geometry must be consistent")

        # Set attributes
        self._geom_type = geometry[0]
        self._gpd_df = gpd_df
        self._point_layer_class = PointLayer
        self._polygon_layer_class = PolygonLayer
        self._line_layer_class = LineLayer
        self.name = name

    @iterate_over_geometry()
    def _explode(self, geometry):
        if type(geometry) == self._multi_geometry_class:
            return list(geometry)

    @iterate_over_geometry()
    def _split(self, geometry, threshold, method, no_multipart, show_progressbar):
        if geometry.__getattribute__(self._split_threshold) > threshold:
            split_geom = self._split_methods[method](geometry, threshold)
            if no_multipart:
                split_geom = explode(split_geom)
            return split_geom

    @return_new_instance
    def add_points_to_geometry(self, distance):
        """ Add points to geometry

        Parameters
        ----------
        distance:

        Returns
        -------
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = [add_points_to_line(geom, distance) for geom in self.exterior]

        return outdf

    @return_new_instance
    def add_z(self, dem, no_data_value=0):
        """ Add z dimension to layer

        Parameters
        ----------
        dem: DigitalElevationModel
            digitial elevation model
        no_data_value: int or float
            value by which NaN must be replaced

        Returns
        -------
        """
        from gistools.raster import DigitalElevationModel

        check_type(dem, DigitalElevationModel)
        check_proj(self.crs, dem.crs)

        geometry = []
        for i, x, y in self.iterxy():
            z = np.array([dem.get_value_at(yi, xi) for xi, yi in zip(x, y)])
            z[np.isnan(z)] = no_data_value
            geometry.append(self.geometry[i].__class__([(xi, yi, zi)
                                                        for xi, yi, zi in zip(x, y, z)]))

        outdf = self._gpd_df.copy()
        outdf.geometry = geometry

        return outdf

    @return_new_instance
    def append(self, other):
        """ Append other GeoLayer to instance


        Parameters
        ----------
        other: GeoLayer
            instance of the same class as self

        Returns
        -------

        """
        if self.__class__ != other.__class__:
            raise GeoLayerError("Must append '%s' but input is '%s'" %
                                (type(self).__name__, type(other).__name__))

        other = other.to_crs(self.crs)

        return self._gpd_df.append(other._gpd_df, ignore_index=True, sort=False)

    @return_new_instance
    def append_attribute(self, **kwargs):
        """ Add new attribute to attribute table

        Add attribute using "attribute name"=value keyword format

        Returns
        -------
        """
        return self._gpd_df.assign(**kwargs)

    @type_assert(distance=(int, float))
    def buffer(self, distance, resolution=16):
        """ Return layer with buffer geometry

        Return buffer zone(s) around object in new layer

        Parameters
        ----------
        distance: radius of the buffer zone
        resolution:

        Returns
        -------
        PolygonLayer
        """
        # Warning: use df.copy() otherwise it passes by reference and modifies the original object !
        return self._polygon_layer_class.from_gpd(self._gpd_df.copy(),
                                                  geometry=self._gpd_df.buffer(distance,
                                                                               resolution),
                                                  crs=self.crs)

    def centroid(self):
        """ Get centroid of geometries

        Returns
        -------
        PointLayer
        """
        return self._point_layer_class.from_gpd(self._gpd_df.copy(),
                                                geometry=self._gpd_df.centroid,
                                                crs=self.crs)

    @return_new_instance
    def dissolve(self, by=None, aggfunc='first', as_index=False):
        """ Dissolve geometry with respect to attribute(s)

        Parameters
        ----------

        by:
        aggfunc:
        as_index:

        Returns
        -------
        """

        return self._gpd_df.dissolve(by=by, aggfunc=aggfunc, as_index=as_index)

    def distance(self, other):
        """ Get min distance to other layer

        Compute min distance to other layer,
        for each feature (no element-wise).
        Return a numpy array

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        """

        return self._distance_and_nearest_neighbor(other)[0]

    def distance_and_nearest_neighbor(self, other):
        """ Get both min distance and nearest neighbor

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        """

        return self._distance_and_nearest_neighbor(other)

    @return_new_instance
    def drop(self, labels=None, axis=0, index=None, attributes=None):
        """ Drop columns/rows according to labels

        Parameters
        ----------
        labels:
        axis:
        index:
        attributes:

        Returns
        -------
        """
        return self._gpd_df.drop(labels, axis, index, columns=attributes)

    @return_new_instance
    def drop_attribute(self, attr_name):
        """ Drop attribute

        Parameters
        ----------
        attr_name: str or list[str]
            attribute name

        Returns
        -------
        """
        attr_name = [attr_name] if isinstance(attr_name, str) else attr_name
        drop_attr = [attr for attr in attr_name if attr in self.attributes()]

        if drop_attr:
            return self._gpd_df.drop(attr_name, axis=1)
        else:
            return self._gpd_df.copy()

    def drop_duplicate_geometries(self):
        """ Drop duplicate geometries

        Use Rtree spatial index and shapely equals method to get duplicate geometries
        """
        duplicate = []

        # Do not use property as we want to delete entries in index
        r_tree = r_tree_idx(self.geometry)

        for n in range(len(self)):
            r_tree.delete(n, self.geometry[n].bounds)
            if is_in_collection(self.geometry[n], self.geometry, r_tree):
                duplicate.append(n)
        return self.drop(index=duplicate)

    @return_new_instance
    def drop_duplicates(self, *args, **kwargs):
        """ Drop duplicates

        Duplicate only consider exactly equal geometries.
        Use "drop_duplicate_geometries" if you want to drop
        topologically equal geometries.
        Thanks to https://github.com/geopandas/geopandas/issues/521#issuecomment-346808004

        Returns
        -------
        GeoLayer
            new instance
        """
        outdf = self._gpd_df.copy()
        # wkb to make geometry hashable
        outdf.geometry = outdf.geometry.apply(lambda geom: geom.wkb)
        outdf = outdf.drop_duplicates(*args, **kwargs)
        outdf.geometry = outdf.geometry.apply(lambda geom: wkb.loads(geom))

        return outdf

    def envelope(self):
        """ Extract rectangular polygon that contains each geometry

        """
        try:
            return self._polygon_layer_class.from_gpd(self._gpd_df.copy(),
                                                      geometry=self._gpd_df.envelope,
                                                      crs=self.crs)
        except GeoLayerError:
            return self._point_layer_class.from_gpd(self._gpd_df.copy(),
                                                    geometry=self._gpd_df.envelope,
                                                    crs=self.crs)

    def explode(self):
        """ Explode "multi" geometry into "single"

        Thanks to https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26
        """
        return self._explode()

    def get_underlying_points_as_new_layer(self, location=None):
        """ Get underlying points constituting the layer as a new PointLayer instance

        Get underlying point coordinates of the layer as a new point layer.
        If location is None, all points are converted

        Parameters
        ----------
        location: list[tuple]
            list of tuples with 2 elements (first: object number,
            second: point index within coordinate list)

        Returns
        -------
        PointLayer
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

    def hausdorff_distance(self, other):
        """ Compute hausdorff distance element-wise

        Parameters
        ----------
        other: GeoLayer or geopandas.GeoSeries or geopandas.GeoDataFrame

        Returns
        -------
        Series
        """
        return Series([geom1.hausdorff_distance(geom2)
                       for geom1, geom2 in zip(self.geometry,
                                               other.geometry)])

    def intersecting_features(self, other):
        """ Which geometry of other layer does intersect ?

        Retrieve which elements of other intersect
        with elements of geo layer (not element-wise)

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        list
        """
        check_type(other, GeoLayer)
        list_of_intersecting_features = []

        for geom in self.geometry:
            list_of_intersecting_features.append(intersecting_features(geom,
                                                                       other.geometry,
                                                                       other.r_tree_idx))

        return list_of_intersecting_features

    def intersects(self, other):
        """ Does layer intersect with other ? (Not element wise)

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        numpy.ndarray
            Array of boolean
        """
        try:
            check_type(other, GeoLayer)
        except TypeError:
            raise GeoLayerError("input must be a geo layer but is: %s" % other.__class__)

        is_intersecting = []
        for geom in self.geometry:
            is_intersecting.append(any(intersects(geom, other.geometry, other.r_tree_idx)))

        return np.asarray(is_intersecting)

    def is_exploded(self):
        """ Test if geometry is "exploded"

        Test whether geometry is exploded, that is
        whether geometry is only single-part and not
        multi-part

        Returns
        -------
        bool
        """
        exploded_geom = explode(self.geometry)
        if len(self) == len(exploded_geom):
            return True
        else:
            return False

    def is_valid(self):
        """ Is geometry valid ?

        Returns
        -------
        bool
        """
        return self._gpd_df.is_valid

    def iterrows(self):
        return self._gpd_df.iterrows()

    def iterxy(self):
        """ Iterate over x and y coordinates of
        all layer's geometries

        Returns
        -------
        tuple[numpy.ndarray]
        """
        num = 0
        while num < len(self):
            yield self.index[num], self.xy(num)[0], self.xy(num)[1]
            num += 1

    def iterxy_in_geometry(self, geometry_id):
        """ Iterate over x and y coords of given geometry

        Parameters
        ----------
        geometry_id: int
            geometry's index
        """
        num = 0
        while num < len(self.xy(geometry_id)[0]):
            yield self.xy(geometry_id)[0][num], self.xy(geometry_id)[1][num]
            num += 1

    def keep_attributes(self, attr_name):
        """ Keep only specific attributes in given layer

        Parameters
        ----------
        attr_name: str or list[str]
            attribute name

        Returns
        -------
        GeoLayer
        """
        attr_name = [attr_name] if isinstance(attr_name, str) else attr_name
        drop_attr = [attr for attr in self.attributes() if attr not in attr_name]

        if attr_name:
            return self.drop_attribute(drop_attr)
        else:
            return self.copy()

    def length_xy_of_geometry(self, geometry_id):
        """ Compute 2D length of given geometry

        Parameters
        ----------
        geometry_id: int
            geometry's index

        Returns
        -------
        numpy.ndarray
            Array of length values
        """
        x, y = np.array(self.xy(geometry_id)[0]), np.array(self.xy(geometry_id)[1])
        return np.sqrt((x[1::] - x[:-1:]) ** 2 + (y[1::] - y[:-1:]) ** 2)

    def length_xyz_of_geometry(self, geometry_id):
        """ Compute 3D length of given geometry

        Parameters
        ----------
        geometry_id: int
            geometry's index

        Returns
        -------
        numpy.ndarray
            Array of length values
        """
        x, y = np.array(self.xy(geometry_id)[0]), np.array(self.xy(geometry_id)[1])
        if self.exterior[geometry_id].has_z:
            z = np.array(self.exterior[geometry_id].coords)[:, 2]
        else:
            z = np.zeros(len(x))

        return np.sqrt((x[1::] - x[:-1:]) ** 2 + (y[1::] - y[:-1:]) ** 2 + (z[1::] - z[:-1:]) ** 2)

    def length_xyz(self):
        """ Compute length with Z dimension

        Returns
        -------
        list[float]
            Collection of length values
        """
        length = []
        for geom in self.exterior:
            if geom.has_z:
                x, y = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
                z = np.array(geom.coords)[:, 2]
                length.append(np.sum(np.sqrt((x[1::] - x[:-1:])**2 + (y[1::] - y[:-1:])**2 +
                                             (z[1::] - z[:-1:])**2)))
            else:
                length.append(geom.length)

        return length

    @return_new_instance
    def merge(self, on=None):
        """ Merge data based on attribute

        Parameters
        ----------
        on:

        Returns
        -------
        GeoLayer
        """
        # TODO: implement method

    def nearest_neighbor(self, other):
        """ Get nearest neighbor in other layer

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        """
        return self._distance_and_nearest_neighbor(other)[1]

    def nearest_neighbors(self, other, buffer_distance):
        """ Get nearest neighbors of other layer

        Get nearest neighbor elements of other layer
        within a specific buffer around each element
        of geo layer

        Parameters
        ----------
        other: GeoLayer
        buffer_distance: float
            Buffer distance around geometries in "other"

        Returns
        -------
        list[GeoLayer]
            list of new layers of the same class as other
        """
        check_proj(self.crs, other.crs)

        buffered_layer = self.buffer(buffer_distance)
        nearest_neighbors = []
        nn_id = []

        for i, buffer_geom in enumerate(buffered_layer.geometry):
            is_intersecting = intersects(buffer_geom, other.geometry, other.r_tree_idx)
            if any(is_intersecting):
                nearest_neighbors.append(other[is_intersecting])
                nn_id.append(other.index[is_intersecting])
            else:
                nearest_neighbors.append(None)
                nn_id.append(None)

        return nearest_neighbors, nn_id

    def nearest_point_in_layer(self, points):
        """ Find nearest points in layer elements

        Find nearest point in layer geometry coordinates.
        Return element row index and index of the point within
        coordinate sequence for each element (row) of the data frame

        Parameters
        ----------
        points: PointLayer

        Returns
        -------
        list
        """

        check_type(points, PointLayer)

        # Get nearest element index in other layer
        nearest_neighbor = points.nearest_neighbor(self)

        # Output
        result = []

        for n, geom in enumerate(points.geometry):
            x, y = np.array(self.exterior[nearest_neighbor[n]].coords.xy[0]), \
                   np.array(self.exterior[nearest_neighbor[n]].coords.xy[1])
            argmin = np.argmin(np.sqrt((geom.x - x) ** 2 + (geom.y - y) ** 2))
            result.append((nearest_neighbor[n], argmin))

        # Careful !! Do not return sorted result as each entry
        # corresponds to point in layer "points" in ascending order
        return result

    @return_new_instance
    def overlay(self, other, how):
        """ Apply overlay geometry operation from another layer (same dimension or higher)

        Parameters
        ----------
        other: GeoLayer
        how: str
            type of overlay geometry operation

        Returns
        -------
        GeoLayer
        """
        check_proj(self.crs, other.crs)

        if isinstance(self, (PointLayer, LineLayer)):
            try:
                check_type(other, (LineLayer, PolygonLayer))
            except TypeError:
                raise GeoLayerError("other must be LineLayer or "
                                    "PolygonLayer but is '%s'" % other.__class__)
        if isinstance(self, PolygonLayer):
            try:
                check_type(other, PolygonLayer)
            except TypeError:
                raise GeoLayerError("other must be PolygonLayer but is '%s'" % other.__class__)

        how = check_string(how, ("intersection", "difference", "union", "symmetric_difference"))
        # TODO: implement "union", "symmetric_difference" and "identity" methods

        if how == "intersection":
            return _intersection(self, other)
        elif how == "difference":
            return _difference(self, other)
        else:
            raise GeoLayerError("'%s' is not a valid overlay method" % how)
        # elif how == "union":
        #     return _union(self, other)
        # else:  # symmetric_difference
        #     return _symmetric_difference(self, other)

    def pairwise_distance(self, other):
        """ Compute distance between all elements from two GeoLayers

        Return the pairwise matrix of distances between
        two geo layers

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        numpy.ndarray
            Pairwise distance matrix
        """
        distance_matrix = np.zeros((len(self), len(other)))

        for m, geom in enumerate(self.geometry):
            for n, other_geom in enumerate(other.geometry):
                distance_matrix[m, n] = geom.distance(other_geom)

        return distance_matrix

    def project(self, other):
        """ Project layer geometry onto other geometry

        Projection results in PointLayer instance with
        points belonging to other geometry and nearest
        to layer objects

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        PointLayer
            Layer with points belonging to other geometry
        """
        nn = self.nearest_neighbor(other)
        outdf = self._gpd_df.copy()

        for idx, n in enumerate(nn):
            list_of_points = MultiPoint(self.exterior[idx].coords)
            distance = [other.geometry[n].distance(point) for point in list_of_points]
            point = list_of_points[np.argmin(distance)]
            dist = other.geometry[n].project(point)
            pt = other.geometry[n].interpolate(dist)
            outdf.loc[idx, 'geometry'] = pt

        return self._point_layer_class(outdf, self.name)

    @return_new_instance
    def rename(self, attribute_name, new_name):
        """ Rename attribute in table

        Parameters
        ----------
        attribute_name: str or list[str]
        new_name: str or list[str]

        Returns
        -------
        GeoLayer
        """
        outdf = self._gpd_df.copy()
        attribute_name = [attribute_name] if isinstance(attribute_name, str) else attribute_name
        new_name = [new_name] if isinstance(new_name, str) else new_name
        if len(attribute_name) != len(new_name):
            raise GeoLayerError("Both inputs must have the same length")

        return outdf.rename(index=str,
                            columns={attr_name: name for attr_name, name in zip(attribute_name,
                                                                                new_name)})

    @return_new_instance
    def simplify(self, tolerance=0.2):
        """ Simplify object geometry

        Parameters
        ----------
        tolerance: float
            Simplification tolerance

        Returns
        -------
        GeoLayer
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = outdf.geometry.simplify(tolerance)
        return outdf

    @return_new_instance
    def sjoin(self, other, op="intersects"):
        """ Spatial join with another layer

        Parameters
        ----------
        other: GeoLayer
        op: str
            spatial operation

        Returns
        -------
        GeoLayer
        """
        # When doing the spatial join, we drop the "index_right" column added by Geopandas
        return gpd.sjoin(self._gpd_df, other._gpd_df, how="left", op=op).drop("index_right", axis=1)

    def split(self, threshold, method=None, no_multipart=None, show_progressbar=False):
        """ Split geometry

        Parameters
        ----------
        threshold: float
            threshold for split operation
        method: str
            method used to split geometry
        no_multipart: bool
            Should resulting geometry be single-part (no multi-part) ?
        show_progressbar: bool
            Show progress bar in console for long iterations

        Returns
        -------
        GeoLayer
        """

        method = check_string(method, self._split_methods.keys())

        return self._split(threshold, method, no_multipart, show_progressbar=show_progressbar)

    @return_new_instance
    def to_2d(self):
        """ Convert 3D geometry to 2D

        Returns
        -------
        GeoLayer
        """
        outdf = self._gpd_df.copy()
        try:
            outdf.geometry = self._gpd_df.geometry.apply(to_2d)
        except TypeError:
            pass

        return outdf

    # TODO: update the following method by using PyRasta
    def to_array(self, geo_grid: GeoGrid, attribute, data_type='uint8', all_touched=False):
        """ Convert layer to numpy array

        Parameters
        ----------
        geo_grid: GeoGrid
            GeoGrid instance
        attribute: str
            valid attribute of GeoLayer dataset
        data_type:
        all_touched: bool
            boolean --> rasterization type (cell centers or "all touched")

        Returns
        -------
        numpy.ndarray
        """
        check_type(geo_grid, GeoGrid, attribute, str)

        if attribute not in self.attributes():
            raise GeoLayerError("%s is not a valid attribute" % attribute)

        return geopandas_to_array(self._gpd_df, attribute, geo_grid, data_type, all_touched)

    # TODO: update the following method by using raster from PyRasta
    def to_raster_map(self, geo_grid: GeoGrid, attribute):
        """ Convert layer to RasterMap instance

        Parameters
        ----------
        geo_grid:
        attribute:

        Returns
        -------
        gistools.raster.RasterMap
        """
        from gistools.raster import RasterMap
        return RasterMap(self.to_array(geo_grid, attribute), geo_grid)

    @return_new_instance
    def to_crs(self, crs=None, epsg=None):
        """ Convert GeoLayer into new CRS

        Parameters
        ----------
        crs: str or pyproj.CRS
            valid pjection (proj4, CRS, etc.)
        epsg: int
            EPSG code

        Returns
        -------
        GeoLayer
        """
        if crs is not None:
            try:
                crs = pyproj.CRS(crs)
            except ValueError:
                raise GeoLayerError("Invalid projection")

        elif epsg is not None and crs is None:
            try:
                crs = pyproj.CRS(epsg)
            except ValueError:
                raise GeoLayerError("Invalid EPSG code")

        else:
            raise GeoLayerError("Must set either crs or epsg code")

        if crs != self.crs:
            return self._gpd_df.to_crs(crs=crs)
        else:
            return self.copy()

    @type_assert(file_path=str)
    def to_csv(self, file_path, attributes=None, *args, **kwargs):
        """ Write layer to csv file

        Parameters
        ----------
        file_path: str
            path to csv file
        attributes: list[str]
            layer attributes to write in csv
        args:
            arguments to be passed on geopandas to-csv method
        kwargs:
            keyword arguments to be passed on geopandas to_csv method
        """
        if attributes is not None:
            check_sub_collection_in_collection(attributes, self.attributes())

        self._gpd_df.to_csv(file_path, columns=attributes, *args, **kwargs)

    def to_file(self, file_path, driver="ESRI Shapefile", **kwargs):
        """ Write geo layer to file

        Parameters
        ----------
        file_path: str
            Valid file path
        driver: str
            valid fiona driver (fiona.supported_drivers)
        """
        import fiona
        driver = check_string(driver, list(fiona.supported_drivers.keys()))
        try:
            os.remove(file_path)
        except OSError:
            pass
        self._gpd_df.to_file(file_path, driver=driver, **kwargs)

    def xy(self, n):
        """ Return xy coords of geo layer nth geometry

        Parameters
        ----------
        n: int
            geometry's index

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
        """
        return self.exterior[n].coords.xy[0], self.exterior[n].coords.xy[1]

    def attributes(self):
        """ Return attributes of geo layer

        Returns
        -------
        """

        # TODO: return only attributes (no geometry)
        columns = [col for col in self._gpd_df.keys() if col != 'geometry']
        return columns
        # return self._gpd_df.keys()

    def plot(self, *args, **kwargs):
        """ Plot GeoLayer

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """

        return plot_geolayer(self, *args, **kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def shallow_copy(self):

        return copy.copy(self)

    def __len__(self):

        return len(self._gpd_df)

    def __repr__(self):
        return repr(self._gpd_df)

    def __geo_interface__(self):
        return self._gpd_df.__geo_interface__

    # __getitem__ method returns new instance or pandas Series or inner value
    @return_new_instance
    def __getitem__(self, key):
        """ Get item from layer

        Parameters
        ----------
        key:

        :Example:
            >>> m = layer[idx]
            return a Series, but
            >>> m = layer[[idx]]
            return a GeoLayer
        """
        try:
            return self._gpd_df[key].copy()  # .copy()
        except KeyError:
            try:
                return self._gpd_df.loc[key].copy()
            except KeyError:
                raise GeoLayerError("Not a valid key/location/index")

    def __setitem__(self, key, value):

        self._gpd_df[key] = value

        return self

    @property
    def boundary(self):
        return self._gpd_df.boundary

    @property
    def bounds(self):
        return self._gpd_df.bounds

    @property
    def total_bounds(self):
        return self._gpd_df.total_bounds

    @property
    def crs(self):
        return self.pyproj.crs

    @property
    def exterior(self):
        return self.geometry

    @property
    def geometry(self):
        return self._gpd_df.geometry

    @property
    def geo_type(self):
        if self.pyproj.crs.is_geographic:
            return "latlon"
        else:
            return "equal"

    @property
    def index(self):
        return self._gpd_df.index

    @index.setter
    def index(self, idx):
        self._gpd_df.index = idx

    @property
    def length(self):
        return self._gpd_df.geometry.length

    @property
    def pyproj(self):
        # Update 12/10/2019: due to "FutureWarning" in pyproj library,
        # use crs['init'] so that CRS initialization
        # method is '<authority>:<code>'
        # if isinstance(self._gpd_df.crs, dict):
        #     crs = self._gpd_df.crs['init']
        # elif isinstance(self._gpd_df.crs, str):
        #     crs = self._gpd_df.crs.replace("+init=", "")
        # else:
        #     crs = self._gpd_df.crs

        return pyproj.Proj(self._gpd_df.crs)

    # Rtree property: lazy property (only computed once when accessed for the first time)
    @lazyproperty
    def r_tree_idx(self):
        return r_tree_idx(self.geometry)

    @property
    def schema(self):
        return infer_schema(self._gpd_df)

    ###################
    # Protected methods

    def _distance_and_nearest_neighbor(self, other):
        """ Get min distance and nearest neighbor of other layer

        Parameters
        ----------
        other: GeoLayer

        Returns
        -------
        """
        check_type(other, GeoLayer)
        check_proj(self.crs, other.crs)

        min_distance = np.zeros(len(self))
        nearest_neighbor = np.zeros(len(self), dtype='int')

        for i, geom in enumerate(self.geometry):
            nearest_neighbor[i], min_distance[i] = nearest_feature(geom, other.geometry,
                                                                   other.r_tree_idx)

        return min_distance, nearest_neighbor

    ############
    # Class methods
    @classmethod
    def from_collection(cls, collection, **kwargs):
        """ Build layer from collection of geometries

        Parameters
        ----------
        collection: tuple or list
        kwargs:
            attributes of corresponding geometries

        Returns
        -------
        GeoLayer
        """
        gpd_df = gpd.GeoDataFrame.from_dict(dict(**kwargs, geometry=collection))

        return cls.from_gpd(gpd_df, crs=gpd_df.crs)

    @classmethod
    def from_gpd(cls, *gpd_args, **kwargs):
        """ Build layer from geopandas GeoDataFrame arguments

        Parameters
        ----------
        gpd_args:
            geopandas arguments
        kwargs:
            geopandas/geolayer keyword arguments

        Returns
        -------
        GeoLayer
        """
        if "name" in kwargs.keys():
            name = kwargs.pop("name")
            layer = cls(gpd.GeoDataFrame(*gpd_args, **kwargs), name=name)
        else:
            layer = cls(gpd.GeoDataFrame(*gpd_args, **kwargs))

        return layer

    # @classmethod
    # def from_osm(cls, place, tag, values=None, by_poly=True, timeout=180):
    #     """ Build layer from OpenStreetMap query
    #
    #     Parameters
    #     ----------
    #     place: str
    #         single place name query (e.g.: "London", "Paris", etc.)
    #     tag: str
    #         OSM tag
    #     values: str or list[str]
    #         str/list of possible values corresponding to OSM tag
    #     by_poly: bool
    #         if True, search within place's corresponding polygon, otherwise use bounds
    #     timeout: float
    #
    #     Returns
    #     -------
    #     GeoLayer
    #     """
        # list_of_gdf = []
        # jsons = download_osm_features(place, cls._osm_type, tag, values, by_poly, timeout)
        # for json in jsons:
        #     list_of_gdf.append(json_to_geodataframe(json, cls._geometry_class.__name__))
        #
        # return cls(gpd.GeoDataFrame(concat(list_of_gdf, ignore_index=True),
        #                             crs=list_of_gdf[0].crs),
        #            name=tag)


class PolygonLayer(GeoLayer):
    """ Polygon layer instance

    Geo layer of polygons
    """

    # _partition_methods = {'area': area_partition_polygon}
    _split_methods = {'katana_simple': katana, 'katana_centroid': katana_centroid,
                      'fishnet': fishnet, 'hexana': hexana}
    _split_threshold = 'area'
    _geometry_class = Polygon
    _multi_geometry_class = MultiPolygon
    _osm_type = 'nwr'

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self._geom_type != 'Polygon':
            raise PolygonLayerError("Geometry must be 'Polygon' "
                                    "but is '{}'".format(self._geom_type))

        # Clean geometries
        self._gpd_df.geometry = self._gpd_df.geometry.buffer(0, 0)

    @iterate_over_geometry()
    def _partition(self, geometry, threshold, disaggregation_factor,
                   precision, recursive, split_method,
                   show_progressbar, **metis_options):
        if geometry.area > threshold:
            return area_partition_polygon(
                geometry, threshold, disaggregation_factor=disaggregation_factor,
                precision=precision, recursive=recursive, split=self._split_methods[
                    split_method], **metis_options)

    def attr_area(self, other, attr_name: str, normalized: bool = False):
        """ Area of attribute from other PolygonLayer in current layer

        Spatial join which gives area of specific attribute value
        in other PolygonLayer intersecting current layer

        Parameters
        ----------
        other: PolygonLayer
        attr_name: str
            Attribute name in other PolygonLayer
        normalized: bool
            is area normalized with respect to each layer's polygon area ?

        Returns
        -------
        """
        check_type(other, PolygonLayer, attr_name, str, normalized, bool)
        check_proj(self.crs, other.crs)

        attr_value = {attr: np.zeros(len(self)) for attr in set(other[attr_name])}
        for i, geom in enumerate(self.geometry):
            area = np.array(shared_area_among_collection(geom, other.geometry,
                                                         normalized, other.r_tree_idx))
            for val, _area in zip(other[attr_name][area > 0], area[area > 0]):
                attr_value[val][i] += _area

        return attr_value

    def clean_geometry(self, delete_invalid: bool = False):
        """ Clean invalid geometries

        Send warning if invalid geometry is removed from layer

        Parameters
        ----------
        delete_invalid: bool
            if True, delete invalid geometries

        Returns
        -------
        PolygonLayer
        """
        layer = self.buffer(0, 0)

        if delete_invalid:
            new_layer = layer[layer.is_valid()]
            if len(new_layer) != len(layer):
                warnings.warn("%d invalid geometries removed in layer '%s'" %
                              (len(layer) - len(new_layer), layer.name),
                              PolygonLayerWarning)
            return new_layer
        else:
            return layer

    @return_new_instance
    def convex_hull(self):
        """ Return convex hull

        Returns
        -------
        PolygonLayer
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = self._gpd_df.convex_hull

        return outdf

    def distance_of_centroid_to_boundary(self):
        """ Return distance (min and max) of centroid to polygon's boundary

        Returns
        -------
        Series
            Series of min and max distance
        """
        min_distance = self.boundary.distance(self._gpd_df.centroid)
        max_distance = self.hausdorff_distance(self._gpd_df.centroid)

        return min_distance, max_distance

    def extract_overlap(self):
        """ Extract internal overlap polygon geometries

        Return as many layers as necessary where geometries do not overlap

        Returns
        -------
        tuple
            Tuple of PolygonLayer instances
        """
        layer = self.copy()
        outlayers = []

        while "there are overlaps to extract":
            r_tree = r_tree_idx(layer.geometry)  # No property as we delete entries from Index
            to_append = []
            for n in range(len(layer)):
                r_tree.delete(n, layer.geometry[n].bounds)
                _, list_of_overlapping_features = overlapping_features(layer.geometry[n],
                                                                       layer.geometry, r_tree)
                if list_of_overlapping_features:
                    to_append.append(n)

            if not to_append:
                outlayers.append(layer)
                break
            else:
                outlayers.append(layer.drop(index=to_append))
                layer = layer[to_append]

        return tuple(outlayers)

    @return_new_instance
    def fix_overlap(self, how):
        """ Fix internal overlaps

        Overlapping features are regarded as features which
        overlap AND contain or are within each other

        Parameters
        ----------
        how: str
            {'intersection', 'difference', 'union'}

        Returns
        -------
        PolygonLayer
        """
        r_tree = r_tree_idx(self.geometry)
        new_geometry = []
        new_rows = []
        list_of_objects = list(range(len(self)))
        while list_of_objects:
            n = list_of_objects.pop(0)
            r_tree.delete(n, self.geometry[n].bounds)
            feature_idx, list_of_overlapping_features = overlapping_features(self.geometry[n],
                                                                             self.geometry, r_tree)
            if list_of_overlapping_features:
                geom_union = cascaded_union([geometry for geometry in list_of_overlapping_features])
                if how == "intersection":
                    geom_result = self.geometry[n].intersection(geom_union)
                elif how == "difference":
                    geom_result = self.geometry[n].difference(geom_union)
                else:  # union
                    geom_result = self.geometry[n].union(geom_union)
                for i, geom in zip(feature_idx, list_of_overlapping_features):
                    list_of_objects.remove(i)
                    r_tree.delete(i, geom.bounds)
            else:
                geom_result = self.geometry[n]

            new_geometry.append(geom_result)
            new_rows.append(self._gpd_df.iloc[n])

        outdf = gpd.GeoDataFrame(columns=self.attributes(), crs=self.crs).append(new_rows)
        outdf.geometry = new_geometry

        return outdf

    def has_overlap(self):
        """ Does layer contain any overlap ?

        Returns
        -------
        bool
        """
        for geom in self.geometry:
            if overlaps(geom, self.geometry, self.r_tree_idx).count(True) > 1:
                return True

        return False

    def intersecting_area(self, other, normalized: bool = False):
        """ Return intersecting area with other layer

        Parameters
        ----------
        other: PolygonLayer
        normalized: bool
            intersecting area normalized with respect to layer area

        Returns
        -------
        numpy.ndarray
        """
        check_type(other, PolygonLayer, normalized, bool)
        check_proj(self.crs, other.crs)

        # Return intersecting matrix
        return np.array([shared_area_among_collection(geom, other.geometry,
                                                      normalized, other.r_tree_idx)
                         for geom in self.geometry])

    @return_new_instance
    def overlay(self, other, how):
        """ Use geopandas overlay method for polygons

        Parameters
        ----------
        other: PolygonLayer
        how: str
            Overlay method

        Returns
        -------
        PolygonLayer
        """
        return gpd.overlay(self._gpd_df, other._gpd_df, how=how)

    def partition(self, threshold, disaggregation_factor=16, precision=100, recursive=False,
                  split_method="hexana", show_progressbar=False, **metis_options):
        """ Split polygon layer into sub-polygons with equal areas

        Split polygons into equal areas using graph partitioning theory

        Parameters
        ----------
        threshold: float
            surface threshold for polygon partitioning
        disaggregation_factor: float or int
            disaggregation before re-aggregating
        precision: float or int
            metric precision for partitioning
        recursive: bool
        split_method: str
            method used to split polygons beforehand
        show_progressbar: bool
            show progress bar if necessary
        metis_options:
            optional arguments specific to METIS partitioning package

        Returns
        -------
        PolygonLayer
        """
        split_method = check_string(split_method, self._split_methods.keys())

        return self._partition(threshold, disaggregation_factor, precision, recursive, split_method,
                               show_progressbar=show_progressbar, **metis_options)

    # TODO: define partition based on raster statistics
    def rpartition(self, raster, nparts, parameter="sum",
                   disaggregation_factor=16, split_method="hexana", **metis_options):
        """ Partition polygons using corresponding raster statistics

        Parameters
        ----------
        raster: pyrasta.raster.Raster
        nparts: int
            number of resulting parts of partition
        parameter: str
            parameter to extract from raster (see ZonalStatistics method's names)
        disaggregation_factor: int or float
            disaggregation
        split_method: str
            method used to split polygons beforehand
        metis_options
            optional arguments specific to METIS partitioning package

        Returns
        -------
        PolygonLayer
        """
        split_method = check_string(split_method, self._split_methods.keys())

    def sampler(self, density=None, count=None, precision=1, surface_threshold=50000000):
        """ Sample random points within polygons

        Parameters
        ----------
        density: int or float
            density of the random points with respect to polygon area
        count: int
            number of random points to generate
        precision: int
            sampling accuracy (default: one point each 1 m²)
        surface_threshold: float
            threshold above which intersection predicate is used

        Returns
        -------
        PointLayer
        """
        @njit()
        def generate_rd_pt(xmin, xmax, ymin, ymax, nb_pts):
            # pts = []
            for n in range(nb_pts):
                yield (xmin + random.random() * (xmax - xmin),
                       ymin + random.random() * (ymax - ymin))
                # pts.append((xmin + random.random() * (xmax - xmin),
                #             ymin + random.random() * (ymax - ymin)))
            # return pts

        if density is None and count is None:
            raise PolygonLayerError("Either density or count must be set")

        points = []

        for poly in self.geometry:
            if density:
                # Use Monte-Carlo principle backwards
                size = math.ceil(density * poly.area / precision *
                                 (poly.bounds[2] - poly.bounds[0]) *
                                 (poly.bounds[3] - poly.bounds[1]) / poly.area)
            else:
                size = math.ceil(count * (poly.bounds[2] - poly.bounds[0]) *
                                 (poly.bounds[3] - poly.bounds[1]) / poly.area)

            # TODO: add distance condition (another kind of method ?)
            if poly.area >= surface_threshold:
                rd_pts = [Point(coords) for coords in generate_rd_pt(poly.bounds[0],
                                                                     poly.bounds[2],
                                                                     poly.bounds[1],
                                                                     poly.bounds[3],
                                                                     size)]
                prep_poly = prep(poly)
                for i in range(len(rd_pts)):
                    if prep_poly.contains(rd_pts[i]):
                        points.append(rd_pts[i])
            else:
                rd_pts = MultiPoint(generate_rd_pt(poly.bounds[0],
                                                   poly.bounds[2],
                                                   poly.bounds[1],
                                                   poly.bounds[3],
                                                   size))
                try:
                    points.extend(rd_pts.intersection(poly))
                except TypeError:
                    points.append(rd_pts.intersection(poly))

        return self._point_layer_class.from_gpd(geometry=points, crs=self.crs)

    def shape_factor(self, convex_hull=True):
        """ Return shape factor series

        Parameters
        ----------
        convex_hull: bool
            if True, use convex hull to compute shape factor

        Returns
        -------
        Series
        """
        return [shape_factor(poly, convex_hull) for poly in self.geometry]

    def split(self, surface_threshold, method="katana_simple",
              no_multipart=False, show_progressbar=False):
        """ Split polygons into layer with respect to surface threshold

        Parameters
        ----------
        surface_threshold: float
            surface threshold
        method: str
            method used to split polygons {'katana_simple', 'katana_centroid', 'hexana'}
        no_multipart: bool
            should resulting geometry be single-part (no multi-part) ?
        show_progressbar: bool
            show progress bar in console for long iterations

        Returns
        -------
        PolygonLayer
        """
        return super().split(surface_threshold, method, no_multipart, show_progressbar)

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

    _osm_type = 'nwr'

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Check geometry
        if self._geom_type != 'Line':
            raise LineLayerError("Geometry of LineLayer must be"
                                 " 'Line' but is '{}'".format(self._geom_type))

    @iterate_over_geometry(replace_by_single=True)
    def _douglas_peucker(self, geometry, tolerance, show_progressbar):
        return LineString(rdp(np.array(geometry.coords), epsilon=tolerance))

    def douglas_peucker(self, tolerance=0, show_progressbar=False):
        """ Apply the Douglas-Peucker algorithm to line geometries

        Parameters
        ----------
        tolerance: float
            tolerance or accuracy in line generalization algorithm
        show_progressbar: bool
            either show progressbar or not

        Returns
        -------
        LineLayer
        """
        return self._douglas_peucker(tolerance, show_progressbar=show_progressbar)

    @return_new_instance
    def linemerge(self, by, method="dissolve"):
        """ Merge lines with respect to attribute

        Use dissolve method and merge results to get merely
        LineString objects

        Parameters
        ----------
        by: str
            name of attribute or list of attribute names
        method: str
            {'dissolve', 'join'} method used for merging lines. Either 'dissolve' from
            GeoPandas library or 'join' from greece.gistools.geometry.

        Returns
        -------
        LineLayer
        """
        outdf = gpd.GeoDataFrame(columns=self.attributes(), crs=self.crs)
        geometry = []

        if method == 'dissolve':
            from shapely.ops import linemerge
            new_df = self._gpd_df.dissolve(by=by, as_index=False)
            for idx, geometry in enumerate(new_df.geometry):
                new_geom = linemerge(geometry)
                try:
                    geometry.extend(new_geom)
                    to_append = [new_df.iloc[idx]] * len(new_geom)
                except TypeError:
                    geometry.append(new_geom)
                    to_append = new_df.iloc[idx]
                outdf = outdf.append(to_append, ignore_index=True)
        elif method == 'join':
            from gistools.geometry import merge
            if isinstance(by, str):
                by = [by]
            set_of_values = set([tuple([row[name] for name in by]) for _, row in self.iterrows()])
            for value in set_of_values:
                true = np.full(len(self), True)
                for attr, val in zip(by, value):
                    true = true & (self[attr] == val)
                new_geom = merge(self._gpd_df.geometry[true].values)
                geometry.extend(new_geom)
                outdf = outdf.append(len(new_geom) * [self._gpd_df[true].iloc[0]],
                                     ignore_index=True)
        else:
            raise ValueError("Invalid method for merging. Must be "
                             "either 'dissolve' or 'join' but is '%s'" % method)

        outdf.geometry = geometry

        return outdf

    def radius_of_curvature_of_geometry(self, geometry_id, method="osculating"):
        """ Compute road's radius of curvature

        Parameters
        ----------
        geometry_id: int
            geometry's ID
        method: str

        Returns
        -------
        numpy.ndarray
        """
        return radius_of_curvature(self.geometry[geometry_id], method=method)

    def slope(self, slope_format="percent"):
        """ Compute 3D line slope

        Parameters
        ----------
        slope_format: str
            Slope format (percent, degrees)

        Returns
        -------
        list
        """
        slope_format = check_string(slope_format, {'degree', 'percent'})
        slope = []
        for geom in self.geometry:
            if geom.has_z:
                if slope_format == "percent":
                    slope.append(100 * (geom.coords[-1][2] - geom.coords[0][2])/geom.length)
                else:
                    slope.append(math.atan((geom.coords[-1][2] -
                                            geom.coords[0][2])/geom.length) * 180/math.pi)
            else:
                slope.append(0)

        return slope

    def slope_of_geometry(self, geometry_id, slope_format="percent", z_spatial_resolution=0):
        """ Compute 3D slope of given geometry

        Parameters
        ----------
        geometry_id: int
            geometry index
        slope_format: str
        z_spatial_resolution: float
            spatial accuracy on Z estimates
            (e.g.: DEM resolution from which Z has been derived)

        Returns
        -------
        numpy.ndarray
        """
        slope_format = check_string(slope_format, {'degree', 'percent'})
        if self.geometry[geometry_id].has_z:
            z = np.array(self.exterior[geometry_id].coords)[:, 2]
            if slope_format == "percent":
                slope = 100 * (z[1::] - z[:-1:])/np.maximum(z_spatial_resolution,
                                                            self.length_xy_of_geometry(geometry_id))
            else:
                slope = \
                    np.arctan((z[1::] - z[:-1:]) / np.maximum(
                        z_spatial_resolution, self.length_xy_of_geometry(geometry_id))) * 180/np.pi
        else:
            slope = np.zeros(len(self.exterior[geometry_id].coords))

        return slope

    def split(self, length_threshold, method="cut", no_multipart=False, show_progressbar=False):
        """ Split lines according to length

        Parameters
        ----------
        length_threshold: float
            length threshold
        method: str
            {'cut', 'cut_'}
        no_multipart: bool
        show_progressbar: bool

        Returns
        -------
        LineLayer
        """
        return super().split(length_threshold, method, no_multipart, show_progressbar)

    def split_at_intersections(self):
        # TODO: split lines with lines. Cut lines at any intersection point of the layer
        pass

    @return_new_instance
    def split_at_points(self, points):
        """ Split lines at given points

        Parameters
        ----------
        points: PointLayer

        Returns
        -------
        LineLayer
        """
        check_type(points, PointLayer)
        outdf = gpd.GeoDataFrame(columns=self._gpd_df.columns, crs=self.crs)
        new_geom = []
        new_rows = []

        for idx, geometry in enumerate(self.geometry):
            _, intersecting_points = intersecting_features(geometry,
                                                           points.geometry,
                                                           points.r_tree_idx)
            if intersecting_points:
                geometry = cut_at_points(geometry, intersecting_points)
                new_geom.extend(geometry)
                new_rows.extend([self._gpd_df.iloc[idx]] * len(geometry))
            else:
                new_geom.append(geometry)
                new_rows.append(self._gpd_df.iloc[idx])

        outdf = outdf.append(new_rows, ignore_index=True)
        outdf.geometry = new_geom

        return outdf

    @return_new_instance
    def split_at_underlying_points(self, location):
        """ Split layer elements at existing coordinates

        Parameters
        ----------
        location: list[tuple]
            list of tuples with 2 elements (object number, index within coordinate
            sequence)

        Returns
        -------
        LineLayer
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

    _osm_type = 'node'

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self._geom_type != 'Point':
            raise PointLayerError("Geometry must be 'Point' but is '{}'".format(self._geom_type))
