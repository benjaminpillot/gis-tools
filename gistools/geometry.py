# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import numpy as np
import networkx as nx

from math import sqrt as msqrt

from numba import njit
from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon, GeometryCollection, Polygon, box, LineString, \
    Point, MultiLineString, JOIN_STYLE
from shapely.ops import cascaded_union, linemerge, unary_union, transform

from gistools.coordinates import r_tree_idx
from gistools.graph import part_graph
from gistools.utils.check.type import is_iterable, type_assert


def add_points_to_line(line, threshold):
    """ Add point coordinates to line geometry

    :param line:
    :param threshold:
    :return:
    """
    return linemerge(cut_(line, threshold))


def aggregate_partitions(polygons, weights, nparts, division,
                         weight_attr, split, recursive, **metis_options):
    """ Aggregate polygons into partitions

    :param polygons: polygons to aggregate
    :param weights: polygons' corresponding weight
    :param nparts: number of partitions
    :param division: list of final relative weights of each partition
    :param weight_attr:
    :param split:
    :param recursive:
    :param metis_options:
    :return:
    """
    if "contig" not in metis_options.keys():
        metis_options["contig"] = False
    graph = polygon_collection_to_graph(polygons, weights, split,
                                        metis_options["contig"], weight_attr)
    tpweights = [(d,) for d in division]
    partition = part_graph(graph, nparts, weight_attr, tpweights, recursive, **metis_options)

    # Return unions of polygons belonging to each part (no multi-polygons)
    return explode([no_artifact_unary_union([polygons[n] for n in part]) for part in partition])


def area_partition_polygon(polygon, unit_area, disaggregation_factor, precision,
                           recursive, split, **metis_options):
    """ Partition polygon into a subset of polygons of equal area

    :param polygon: polygon intended to be partitioned
    :param unit_area: area of a sub-polygon
    :param disaggregation_factor: factor use to discretize polygons before aggregation
    :param recursive: k-way or recursive method for partitioning
    :param precision: metric precision for sub-polygon area
    :param split: function used to split polygon into smaller unit blocks
    :param metis_options: specific METIS options (see METIS manual)
    :return:
    """
    nparts = int(polygon.area/unit_area)

    if nparts <= 1 and (polygon.area - unit_area) < unit_area/disaggregation_factor:
        return [polygon]

    # Split polygon into sub-elements
    split_poly = split_polygon(polygon, split, unit_area/disaggregation_factor, get_explode=True)

    division = [unit_area/polygon.area] * nparts
    if polygon.area % unit_area != 0:  # and (polygon.area - nparts * unit_area) >= unit_area/disaggregation_factor:
        division += [(polygon.area - nparts * unit_area)/polygon.area]
        nparts += 1

    area = [int(poly.area / precision) for poly in split_poly]

    return aggregate_partitions(split_poly, area, nparts, division, "area",
                                split, recursive, **metis_options)


def centroid(point_collection):
    """ Retrieve centroid of multiple points

    :param point_collection:
    :return:
    """
    x_centroid = np.mean([pt.x for pt in point_collection])
    y_centroid = np.mean([pt.y for pt in point_collection])

    return Point([x_centroid, y_centroid])


def connect_lines_to_point(line_collection, point):
    """ Connect a set of lines to some point

    :param line_collection:
    :param point:
    :return:
    """
    new_line_collection = []
    for line in line_collection:
        if Point(line.coords[0]).distance(point) < Point(line.coords[-1]).distance(point):
            new_line_collection.append(LineString(point.coords[:] + line.coords[:]))
        else:
            new_line_collection.append(LineString(line.coords[:] + point.coords[:]))

    return new_line_collection


def cut(line, threshold, count=0):
    """ Cut a line in segments

    Cut a line in segments whose length
    is below a threshold value. This method
    is more randomless regarding the final
    size of the line segments. See 'cut_'
    function for more accuracy
    :param line:
    :param threshold:
    :param count:
    :return:
    """
    result = []
    if threshold < 0 or threshold >= line.length or count == 250:
        return [line]
    # Recursively cut line in 2 at midpoint
    p = line.interpolate(0.5, normalized=True)
    split_line = cut_at_point(line, p)
    for sub_line in split_line:
        result.extend(cut(sub_line, threshold, count + 1))

    return result


def cut_(line, threshold):
    """ Cut a line in segments (method 2)

    This method cuts a line in as many segments as necessary,
    depending on the given threshold. For instance, a line
    of 105m will be cut into 10 pieces of 10m + 1 piece of 5m
    if threshold=10
    :param line: LineString
    :param threshold: minimum sub line piece size
    :return:
    """
    if threshold < 0 or threshold >= line.length:
        return [line]

    result = []

    while "It remains line to cut":
        split_line = cut_at_distance(line, threshold/line.length, normalized=True)
        result.append(split_line[0])

        if split_line[1].length > threshold:
            line = split_line[1]
        else:
            result.append(split_line[1])
            break

    return result


def cut_at_distance(line, distance, normalized=False):
    """ Cut line at given distance from starting point

    :param line:
    :param distance:
    :param normalized:
    :return:
    """
    if normalized:
        length = 1
    else:
        length = line.length

    if distance <= 0.0 or distance >= length:
        return [line]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p), normalized=normalized)
        if pd == distance:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
        elif pd > distance:
            cp = line.interpolate(distance, normalized=normalized)
            try:
                return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]
            except ValueError:
                return [LineString(coords[:i] + [(cp.x, cp.y, cp.z)]), LineString([(cp.x, cp.y, cp.z)] + coords[i:])]


def cut_at_point(line, point):
    """ Cut line at point

    Cut line at point, which can be within
    or without the geometry
    :param line:
    :param point:
    :return:
    """
    d = line.project(point)
    return cut_at_distance(line, d)


def cut_at_points(line, points):
    """ Cut line at multiple points

    :param line:
    :param points:
    :return:
    """
    cut_line = []
    distance = [line.project(point) for point in points]
    sorted_points = [point for _, point in sorted(zip(distance, points))]

    for idx, point in enumerate(sorted_points):
        cut_line.extend(cut_at_point(line, point))
        if idx < len(sorted_points) - 1:
            line = cut_line.pop()

    return cut_line


def dissolve(geometry_collection):
    """ Recursively join contiguous geometries in collection

    :param geometry_collection:
    :return:
    """
    if not is_iterable(geometry_collection):
        raise TypeError("Input must be a collection but is '{}'".format(type(geometry_collection)))

    while "There is still geometries to aggregate":

        joint = []
        idx = r_tree_idx(geometry_collection)
        geom_idx = []
        increment = 0

        while len(geom_idx) < len(geometry_collection):

            if increment not in geom_idx:
                geom = geometry_collection[increment]
                union_idx, union = intersecting_features(geom, geometry_collection, idx)

                if len(union) > 0:
                    joint.append(cascaded_union(union))

                for ix in union_idx:
                    idx.delete(ix, geometry_collection[ix].bounds)

                geom_idx.extend(union_idx)

            increment += 1

        if len(joint) < len(geometry_collection):
            geometry_collection = joint
        else:
            break

    return joint


def explode(geometry_collection):
    """ Convert multi-part geometry collection into single-part

    :param geometry_collection: valid geometry collection
    :return:
    """
    single = []
    if not is_iterable(geometry_collection):
        geometry_collection = [geometry_collection]

    for geom in geometry_collection:
        try:
            single.extend(geom)
        except TypeError:
            single.append(geom)

    return single


def fishnet(polygon, threshold):
    """ Intersect polygon with a regular grid or "fishnet"

    :param polygon:
    :param threshold:
    :return:
    """
    return polygon_to_mesh(polygon, threshold, mesh)


def hexana(polygon, threshold):
    """ Split a polygon using a honeycomb grid

    :param polygon: original polygon to split
    :param threshold: unit hexagon surface
    :return: list of polygons
    """
    return polygon_to_mesh(polygon, threshold, honeycomb)


# Thanks to https://gist.github.com/urschrei/17cf0be92ca90a244a91
@njit()
def honeycomb_nb(startx, starty, endx, endy, radius):
    """
    Calculate a grid of hexagon coordinates of the given radius
    given lower-left and upper-right coordinates
    Returns a list of lists containing 6 tuples of x, y point coordinates
    These can be used to construct valid regular hexagonal polygons

    - update 04/23/2019:
        * can give either radius or area of unit hexagon
        * return a list of shapely Polygon

    You will probably want to use projected coordinates for this
    """

    # calculate side length given radius
    sl = (2 * radius) * np.tan(np.pi / 6)
    # calculate radius for a given side-length
    # (a * (math.cos(math.pi / 6) / math.sin(math.pi / 6)) / 2)
    # see http://www.calculatorsoup.com/calculators/geometry-plane/polygon.php

    # calculate coordinates of the hexagon points
    # sin(30)
    p = sl * 0.5
    b = sl * np.cos(np.radians(30))
    w = b * 2
    h = 2 * sl

    # offset start and end coordinates by hex widths and heights to guarantee coverage
    startx = startx - w
    starty = starty - h
    endx = endx + w
    endy = endy + h
    origx = startx

    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    row = 1

    while starty < endy:
        if row % 2 == 0:
            startx = origx + xoffset
        else:
            startx = origx
        while startx < endx:
            p1x = startx
            p1y = starty + p
            p2x = startx
            p2y = starty + (3 * p)
            p3x = startx + b
            p3y = starty + h
            p4x = startx + w
            p4y = starty + (3 * p)
            p5x = startx + w
            p5y = starty + p
            p6x = startx + b
            p6y = starty
            poly = [
                (p1x, p1y),
                (p2x, p2y),
                (p3x, p3y),
                (p4x, p4y),
                (p5x, p5y),
                (p6x, p6y),
                (p1x, p1y)]
            yield poly
            startx += w
        starty += yoffset
        row += 1


def honeycomb(startx, starty, endx, endy, radius=None, area=None):
    """

    Parameters
    ----------
    startx
    starty
    endx
    endy
    radius
    area

    Returns
    -------

    """

    if not radius:
        radius = msqrt(area / (2*msqrt(3)))

    return (Polygon(poly) for poly in honeycomb_nb(startx, starty, endx, endy, radius))


def intersecting_features(geometry, geometry_collection, r_tree=None):
    """ Return list of geometries intersecting with given geometry

    :param geometry:
    :param geometry_collection:
    :param r_tree: rtree index corresponding to geometry collection
    :return:
    """
    is_intersecting = intersects(geometry, geometry_collection, r_tree)
    return [i for i in range(len(geometry_collection)) if is_intersecting[i]], \
           [geom for i, geom in enumerate(geometry_collection) if is_intersecting[i]]


def intersects(geometry, geometry_collection, r_tree=None):
    """ Return if geometry intersects with geometries of collection

    Use this function with large geometry collections
    :param geometry:
    :param geometry_collection:
    :param r_tree:
    :return: list of boolean of length = length(geometry_collection)
    """
    # Use Rtree to speed up !
    if r_tree is None:
        r_tree = r_tree_idx(geometry_collection)

    list_of_intersecting_features = list(r_tree.intersection(geometry.bounds))

    return [False if f not in list_of_intersecting_features else geometry.intersects(geometry_collection[f]) for f in
            range(len(geometry_collection))]


def is_in_collection(geometry, geometry_collection, r_tree):
    """ Test if geometry is present in collection (using shapely 'equals' method)

    :param geometry:
    :param geometry_collection:
    :param r_tree:
    :return:
    """
    _, list_of_intersecting_features = intersecting_features(geometry, geometry_collection, r_tree)
    for geom in list_of_intersecting_features:
        if geometry.equals(geom):
            return True

    return False


def is_line_connected_to(line, geometry_collection):
    """ Is line connected to one of the geometries in collection ?

    :param line:
    :param geometry_collection:
    :return:
    """

    return [other.intersects(Point(line.coords[0])) for other in geometry_collection], [other.intersects(Point(
        line.coords[-1])) for other in geometry_collection]


def katana(polygon, threshold, count=0):
    """ Split a polygon

    See https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/

    Copyright (c) 2016, Joshua Arnott

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
    disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
    following disclaimer in the documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
    GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param polygon: Shapely polygon
    :param threshold:
    :param count:
    :return:
    """
    if count == 0:
        if not polygon.is_valid:
            polygon = polygon.buffer(0, 0)

    result = []
    width = polygon.bounds[2] - polygon.bounds[0]
    height = polygon.bounds[3] - polygon.bounds[1]
    if width * height <= threshold or count == 250:
        return [polygon]
    if height >= width:
        a = box(polygon.bounds[0], polygon.bounds[1], polygon.bounds[2], polygon.bounds[1] + height/2)
        b = box(polygon.bounds[0], polygon.bounds[1] + height/2, polygon.bounds[2], polygon.bounds[3])
    else:
        a = box(polygon.bounds[0], polygon.bounds[1], polygon.bounds[0] + width/2, polygon.bounds[3])
        b = box(polygon.bounds[0] + width/2, polygon.bounds[1], polygon.bounds[2], polygon.bounds[3])

    for sword in (a, b,):
        split_poly = polygon.intersection(sword)
        if not isinstance(split_poly, GeometryCollection):
            split_poly = [split_poly]
        for sub_poly in split_poly:
            if isinstance(sub_poly, (Polygon, MultiPolygon)):
                result.extend(katana(sub_poly, threshold, count+1))

    return result


def katana_centroid(polygon, threshold, count=0):
    """ Split a polygon in equal areas

    Thanks to https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/ and
    Daniel Harasty in http://community-gispython-org-community-projects.955323.n3.nabble.com/Community-Spliting-a
    -polygon- into-two-polygons-with-the-same-area-td4024026.html#a4024033, we merge here both approaches to split a
    polygon into a number of sub-polygons of almost equal areas.
    :param polygon: Shapely polygon
    :param threshold:
    :param count:
    :return:
    """
    if count == 0:
        if not polygon.is_valid:
            polygon = polygon.buffer(0, 0)

    result = []
    width = polygon.bounds[2] - polygon.bounds[0]
    height = polygon.bounds[3] - polygon.bounds[1]
    if width * height <= threshold or count == 250:
        return [polygon]
    if height >= width:
        a = box(polygon.bounds[0], polygon.bounds[1], polygon.bounds[2], polygon.centroid.y)
        b = box(polygon.bounds[0], polygon.centroid.y, polygon.bounds[2], polygon.bounds[3])
    else:
        a = box(polygon.bounds[0], polygon.bounds[1], polygon.centroid.x, polygon.bounds[3])
        b = box(polygon.centroid.x, polygon.bounds[1], polygon.bounds[2], polygon.bounds[3])

    for sword in (a, b,):
        split_poly = polygon.intersection(sword)
        if not isinstance(split_poly, GeometryCollection):
            split_poly = [split_poly]
        for sub_poly in split_poly:
            if isinstance(sub_poly, (Polygon, MultiPolygon)):
                result.extend(katana_centroid(sub_poly, threshold, count+1))

    return result


def length_of_segments(line):
    """ Retrieve segment length in line

    :param line:
    :return:
    """
    return np.diff([line.project(Point(p)) for p in line.coords])


def mask(polygon_collection, mask_collection, fast_intersection_surface):
    """ Geometry mask

    :param polygon_collection:
    :param mask_collection:
    :param fast_intersection_surface:
    :return:
    """

    # Retrieve base layer and mask geometry, split it for faster intersection
    # and explode it (to be sure there is no multi-parts)
    geometry = split_polygon_collection(polygon_collection, fast_intersection_surface, get_explode=True)
    mask_geometry = split_polygon_collection(mask_collection, fast_intersection_surface, get_explode=True)

    # Use Rtree to speed up !
    idx = r_tree_idx(mask_geometry)

    # 0. Initialization
    result = []

    for geom in geometry:
        list_of_intersecting_mask = list(idx.intersection(geom.bounds))
        within = [geom.within(mask_geometry[n]) for n in list_of_intersecting_mask]
        if not any(within):
            is_intersecting = [geom.intersects(mask_geometry[n]) for n in list_of_intersecting_mask]
            if any(is_intersecting):
                difference = geom.difference(cascaded_union([mask_geometry[n] for n in list_of_intersecting_mask]))
                if not difference.is_empty:
                    result.append(difference)
            else:
                result.append(geom)

    # Multi to single + dissolve coincident polygons
    result = explode(result)
    result = [no_artifact_unary_union(poly) for poly in dissolve(result)]

    return result


def merge(line_collection):
    """ Merge connected lines

    :param line_collection:
    :return:
    """
    # Merge MultiLinestring objects returned by the "join" function
    merged_line = [linemerge(line) if isinstance(line, MultiLineString) else line for line in dissolve(line_collection)]

    # Keep only single parts
    return explode(merged_line)


def mesh(startx, starty, endx, endy, side=None, area=None):
    """ Compute a mesh grid

    :param startx:
    :param starty:
    :param endx:
    :param endy:
    :param side:
    :param area:
    :return:
    """
    if not side:
        side = msqrt(area)

    startx = startx - side/2
    starty = starty - side/2
    endx = endx + side/2
    endy = endy + side/2
    origx = startx

    polygons = []
    while starty < endy:
        startx = origx
        while startx < endx:
            poly = [
                (startx, starty),
                (startx, starty + side),
                (startx + side, starty + side),
                (startx + side, starty)]
            polygons.append(Polygon(poly))
            startx += side
        starty += side

    return polygons


def nearest_feature(geometry, geometry_collection, r_tree=None):
    """ Return nearest feature from geometry collection to given geometry

    If some of the geometries intersect, the nearest feature is the one whose centroid is the closest to the centroid
    of the given geometry (but distance remains 0)
    :param geometry:
    :param geometry_collection:
    :param r_tree: rtree index corresponding to geometry collection
    :return: nearest feature index and corresponding distance
    """
    # Use Rtree to speed up !
    if r_tree is None:
        r_tree = r_tree_idx(geometry_collection)

    # Look if some geometries intersect
    list_of_intersecting_features, _ = intersecting_features(geometry, geometry_collection, r_tree)

    if list_of_intersecting_features:
        distance = [geometry.centroid.distance(geometry_collection[n].centroid) for n in list_of_intersecting_features]
        return list_of_intersecting_features[np.argmin(distance)], 0
    else:
        list_of_nearest_features = list(r_tree.nearest(geometry.bounds, 1))
        distance = [geometry.distance(geometry_collection[n]) for n in list_of_nearest_features]
        return list_of_nearest_features[np.argmin(distance)], np.min(distance)


def no_artifact_unary_union(geoms, eps=0.00001):
    """ Make unary union that does not return artifacts

    Thanks to https://gis.stackexchange.com/questions/277334/shapely-polygon-union-results-in-strange-artifacts-of
    -tiny-non-overlapping-area
    :param geoms: list of geoms to aggregate
    :param eps: buffering precision
    :return:
    """
    return unary_union(geoms).buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)


def overlapping_features(geometry, geometry_collection, r_tree=None):
    """ Return list of geometries overlapping with given geometry

    Overlapping geometry is either overlapping in the shapely way,
    or within or containing the other geometry
    :param geometry:
    :param geometry_collection:
    :param r_tree:
    :return:
    """
    idx, list_of_intersecting_features = intersecting_features(geometry, geometry_collection, r_tree)
    _overlaps = [[i, geom] for i, geom in zip(idx, list_of_intersecting_features) if geom.overlaps(geometry) or
                 geom.within(geometry) or geom.contains(geometry)]

    return [overlap[0] for overlap in _overlaps], [overlap[1] for overlap in _overlaps]


def overlaps(geometry, geometry_collection, r_tree=None):
    """ Return if geometry overlaps with geometries of collection

    Overlapping is regarded as any area shared by two geometries
    :param geometry:
    :param geometry_collection:
    :param r_tree:
    :return:
    """
    is_intersecting = intersects(geometry, geometry_collection, r_tree)
    return [False if not is_intersecting[i] else geom.overlaps(geometry) or geom.within(geometry) or geom.contains(
        geometry) for i, geom in enumerate(geometry_collection)]


def polygon_to_mesh(polygon, threshold, method):
    """

    :param polygon:
    :param threshold:
    :param method: {'hexana', 'fishnet'}
    :return:
    """
    grid = method(*polygon.bounds, area=threshold)
    split = []
    for unit in grid:
        if unit.within(polygon):
            split.append(unit)
        elif unit.overlaps(polygon):
            split.append(unit.intersection(polygon))

    return explode(split)


def polygon_collection_to_graph(polygon_collection, weights, split, is_contiguous, weight_attr="weight"):
    """ Convert collection of polygons to networkx graph

    Conversion of a polygon collection into a graph allows
    later graph partitioning
    :param polygon_collection:
    :param weights: weight of each polygon in collection
    :param split: split function
    :param is_contiguous: True or False (metis options)
    :param weight_attr: name of weight attribute
    :return:
    """
    if not is_iterable(polygon_collection):
        raise TypeError("Input must be a collection but is '{}'".format(type(polygon_collection)))

    if 'katana' in split.__name__:
        is_katana = True
    else:
        is_katana = False

    r_tree = r_tree_idx(polygon_collection)
    graph = nx.Graph()

    for n, polygon in enumerate(polygon_collection):
        list_of_intersecting_features, _ = intersecting_features(polygon, polygon_collection, r_tree)
        list_of_intersecting_features.remove(n)
        if list_of_intersecting_features or not is_contiguous:
            if is_katana:
                graph.add_edges_from([(n, feature) for feature in list_of_intersecting_features
                                      if not isinstance(polygon.intersection(polygon_collection[feature]), Point)])
            else:
                graph.add_edges_from([(n, feature) for feature in list_of_intersecting_features])
            graph.add_node(n, **{weight_attr: weights[n]})

    return graph


def radius_of_curvature(line, method="osculating"):
    """ Compute curvature radius of LineString

    :param line:
    :param method: method for computing radius of curvature {'circumscribe', 'osculating'}
    :return:
    """
    def norm(xx, yy):
        return np.sqrt(xx ** 2 + yy ** 2)

    def tangent_vector(xi, yi):
        return (xi[2::] - xi[:-2]) / norm(xi[2::] - xi[:-2], yi[2::] - yi[:-2]), \
               (yi[2::] - yi[:-2]) / norm(xi[2::] - xi[:-2], yi[2::] - yi[:-2])

    if method == "osculating":

        if len(line.coords) >= 3:
            x = np.array(line.coords.xy[0])
            y = np.array(line.coords.xy[1])
            xi1 = np.concatenate((x[1::], [x[-1]]))
            yi1 = np.concatenate((y[1::], [y[-1]]))
            xi_1 = np.concatenate(([x[0]], x[:-1]))
            yi_1 = np.concatenate(([y[0]], y[:-1]))

            tangent_vector_xi1, tangent_vector_yi1 = tangent_vector(xi1, yi1)
            tangent_vector_xi_1, tangent_vector_yi_1 = tangent_vector(xi_1, yi_1)

            coefficient_of_curvature = \
                norm(tangent_vector_xi1 - tangent_vector_xi_1, tangent_vector_yi1 - tangent_vector_yi_1) /\
                norm(x[2::] - x[:-2], y[2::] - y[:-2])
            coefficient_of_curvature[coefficient_of_curvature == 0] = 1e-6
            rad_of_curvature = 1 / coefficient_of_curvature
        else:
            return np.array([10000])

    elif method == "circumscribe":

        segment_length = length_of_segments(line)
        a, b = segment_length[:-1:], segment_length[1::]
        c = []
        if len(line.coords) > 3:
            length_2_by_2_start = length_of_segments(LineString(line.coords[::2]))
            length_2_by_2_end = length_of_segments(LineString(line.coords[1::2]))

            for n in range(len(length_2_by_2_end)):
                c.extend([length_2_by_2_start[n], length_2_by_2_end[n]])

            if len(length_2_by_2_start) > len(length_2_by_2_end):
                c.append(length_2_by_2_start[-1])

        elif len(line.coords) == 3:
            c = LineString(line.coords[::2]).length

        elif len(line.coords) < 3:
            return np.array([10000])

        heron = (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)
        heron[heron < 0] = 0
        divider = np.sqrt(heron)
        divider[divider == 0] = 0.1
        rad_of_curvature = a * b * c / divider

    else:
        rad_of_curvature = []

    # Return values and add replicate to beginning of array (as result of curvature computation returns an array with
    #  length = length(line.coords) - 2): return array with length = length(line.coords) - 1
    return np.concatenate(([rad_of_curvature[0]], rad_of_curvature))


def shape_factor(polygon, convex_hull):
    """ Compute shape factor of given polygon

    Compute shape factor (here, circularity) of
    a given polygon using either convex hull or not
    :param polygon:
    :param convex_hull: should convex hull be used for computing shape ? (bool)
    :return:
    """

    if convex_hull:
        return 4 * np.pi * polygon.convex_hull.area / (polygon.convex_hull.length ** 2)
    else:
        return 4 * np.pi * polygon.area / (polygon.length ** 2)


@type_assert(polygon1=(Polygon, MultiPolygon), polygon2=(Polygon, MultiPolygon), normalized=bool)
def shared_area(polygon1, polygon2, normalized=False):
    """ Get area shared by 2 polygons

    :param polygon1:
    :param polygon2:
    :param normalized:
    :return:
    """
    if not polygon1.intersects(polygon2):
        return 0
    else:
        new_poly = polygon1.intersection(polygon2)
        if normalized:
            return new_poly.area / polygon1.area
        else:
            return new_poly.area


@type_assert(polygon=(Polygon, MultiPolygon), normalized=bool)
def shared_area_among_collection(polygon: Polygon, polygon_collection, normalized: bool = False, r_tree=None):
    """ Get area shared by a polygon with polygons from a collection

    :param polygon:
    :param polygon_collection:
    :param normalized:
    :param r_tree:
    :return:
    """
    if not is_iterable(polygon_collection):
        raise TypeError("Input 2 must be a collection but is '{}'".format(type(polygon_collection)))

    poly_intersects = intersects(polygon, polygon_collection, r_tree)

    return [shared_area(polygon, poly, normalized) if poly_intersects[n] else 0 for n, poly in enumerate(
            polygon_collection)]


def split_collection(geometry_collection, threshold, method, get_explode):
    """ Split geometry collection

    :param geometry_collection:
    :param threshold:
    :param method:
    :param get_explode:
    :return:
    """
    if not is_iterable(geometry_collection):
        raise TypeError("Geometry must be a collection")

    new_collection = []

    for geom in geometry_collection:
        try:
            new_collection.extend(method(geom, threshold))
        except TopologicalError:
            new_collection.append(geom)

    if get_explode:
        new_collection = explode(new_collection)

    # Return new collection
    return new_collection


def split_line_collection(line_collection, threshold, method="cut", get_explode=False):
    """

    :param line_collection:
    :param threshold:
    :param method:
    :param get_explode:
    :return:
    """
    split_method = {'cut': cut}

    return split_collection(line_collection, threshold, split_method[method], get_explode)


def split_polygon(polygon, method, threshold, get_explode):
    """ Split polygon with respect to method

    Split polygon and return exploded (no multi part) if necessary
    :param polygon:
    :param method:
    :param threshold:
    :param get_explode: (boolean) return exploded collection
    :return:
    """
    sp_poly = method(polygon, threshold)
    if get_explode:
        return explode(sp_poly)
    else:
        return sp_poly


def split_polygon_collection(polygon_collection, threshold, method="katana", get_explode=False):
    """ Split a collection of polygons

    :param polygon_collection: collection of shapely polygons
    :param threshold: threshold surface under which no more splitting must be achieved
    :param method: method used for splitting
    :param get_explode:
    :return: new polygon collection with only Polygon geometries (no MultiPolygon geometries)
    """
    split_method = {'katana': katana, 'katana_centroid': katana_centroid}

    return split_collection(polygon_collection, threshold, split_method[method], get_explode)


def to_2d(geometry):
    """ Convert 3D geometry to 2D

    Credit to @feenster and @hunt3ri from
    https://github.com/hotosm/tasking-manager/blob/master/server/services/grid/grid_service.py
    :param geometry:
    :return:
    """
    def _to_2d(x, y, z):
        return tuple(filter(None, [x, y]))

    return transform(_to_2d, geometry)
