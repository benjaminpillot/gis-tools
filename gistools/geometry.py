# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'

import numpy as np
import networkx as nx

from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon, GeometryCollection, Polygon, box, LineString, \
    Point, MultiLineString, JOIN_STYLE
from shapely.ops import cascaded_union, linemerge, unary_union

from gistools.coordinates import r_tree_idx
from gistools.graph import part_graph
from gistools.utils.check.type import is_iterable, type_assert

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def add_points_to_line(line, threshold):
    """ Add point coordinates to line geometry

    :param line:
    :param threshold:
    :return:
    """
    return linemerge(cut_(line, threshold))


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


# Thanks to https://gist.github.com/urschrei/17cf0be92ca90a244a91
def honeycomb(startx, starty, endx, endy, radius=None, area=None):
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
    if not radius:
        radius = np.sqrt(area / (2*np.sqrt(3)))

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

    polygons = []
    row = 1
    counter = 0

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
            polygons.append(Polygon(poly))
            counter += 1
            startx += w
        starty += yoffset
        row += 1

    return polygons


def fishnet(polygon, threshold):
    """ Intersect polygon with a regular grid or "fishnet"

    :param polygon:
    :param threshold:
    :return:
    """
    # TODO: implement fishnet split operation
    pass


def hexana(polygon, threshold):
    """ Split a polygon using a honeycomb grid

    :param polygon: original polygon to split
    :param threshold: unit hexagon surface
    :return: list of polygons
    """
    honey_grid = honeycomb(*polygon.bounds, area=threshold)
    hexa_split = []
    for hexagon in honey_grid:
        if hexagon.within(polygon):
            hexa_split.append(hexagon)
        elif hexagon.overlaps(polygon):
            hexa_split.append(hexagon.intersection(polygon))

    return explode(hexa_split)


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


def is_line_connected_to(line, geometry_collection):
    """ Is line connected to one of the geometries in collection ?

    :param line:
    :param geometry_collection:
    :return:
    """

    return [other.intersects(Point(line.coords[0])) for other in geometry_collection], [other.intersects(Point(
        line.coords[-1])) for other in geometry_collection]


def join(geometry_collection):
    """ Join contiguous geometries in collection

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

        # TODO: use "intersecting_features" function
        while len(geom_idx) < len(geometry_collection):

            if increment not in geom_idx:
                geom = geometry_collection[increment]
                list_of_intersecting_features = list(idx.intersection(geom.bounds))
                list_of_truly_intersecting_features = [n for n in list_of_intersecting_features if
                                                       geom.intersects(geometry_collection[n])]
                union = [geometry_collection[n] for n in list_of_truly_intersecting_features]

                if len(union) > 0:
                    # TODO: use "no_artifact_unary_union" function
                    joint.append(cascaded_union(union))

                for ix in list_of_truly_intersecting_features:
                    idx.delete(ix, geometry_collection[ix].bounds)

                geom_idx.extend(list_of_truly_intersecting_features)

            increment += 1

        if len(joint) < len(geometry_collection):
            geometry_collection = joint
        else:
            break

    return joint


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

    # Retrieve base layer and mask geometry, split it for faster intersection (in 2-km² sub polygons)
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

    # No multipolygons and join overlapping ones
    # result = explode(result)
    result = join(result)
    result = explode(result)

    return result


def merge(line_collection):
    """ Merge connected lines

    :param line_collection:
    :return:
    """
    # Merge MultiLinestring objects returned by the "join" function
    merged_line = [linemerge(line) if isinstance(line, MultiLineString) else line for line in join(line_collection)]

    # Keep only single parts
    return explode(merged_line)


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


def area_partition_polygon(polygon, unit_area, disaggregation_factor, precision, recursive, split="katana_centroid",
                           **metis_options):
    """ Partition polygon into a subset of polygons of equal area

    :param polygon: polygon intended to be partitioned
    :param unit_area: area of a sub-polygon
    :param disaggregation_factor: factor use to discretize polygons before aggregation
    :param recursive: k-way or recursive method for partitioning
    :param precision: metric precision for sub-polygon attributes (area, length, etc.)
    :param split: how to split the blocks composing the partitions ("katana", "hexana")
    :param metis_options: specific METIS options (see METIS manual)
    :return:
    """
    nparts = int(polygon.area/unit_area)

    if nparts <= 1 and (polygon.area - unit_area) < unit_area/disaggregation_factor:
        return [polygon]

    # Split polygon into sub-elements
    if split == "katana_simple":
        split_polygon = katana(polygon, unit_area / disaggregation_factor)
    elif split == "katana_centroid":
        split_polygon = katana_centroid(polygon, unit_area / disaggregation_factor)
    else:
        split_polygon = hexana(polygon, unit_area / disaggregation_factor)

    division = [unit_area/polygon.area] * nparts
    if polygon.area % unit_area != 0:
        division += [(polygon.area - nparts * unit_area)/polygon.area]
        nparts += 1

    area = [int(poly.area / (precision ** 2)) for poly in split_polygon]

    return aggregate_partitions(split_polygon, area, nparts, division, "area", split, recursive, **metis_options)


def aggregate_partitions(polygons, weights, nparts, division, weight_attr, original_split, recursive, **metis_options):
    """ Aggregate polygons into partitions

    :param polygons: polygons to aggregate
    :param weights: polygons' corresponding weight
    :param nparts: number of partitions
    :param division: list of final relative weights of each partition
    :param weight_attr:
    :param original_split:
    :param recursive:
    :param metis_options:
    :return:
    """
    if "contig" in metis_options.keys():
        is_contiguous = metis_options["contig"]
    else:
        is_contiguous = False
    graph = polygon_collection_to_graph(polygons, weights, original_split, is_contiguous, weight_attr)
    tpweights = [(d,) for d in division]
    partition = part_graph(graph, nparts, weight_attr, tpweights, recursive, **metis_options)

    partition_collection = []
    for part in partition:
        partition_collection.append(no_artifact_unary_union([polygons[n] for n in part]))

    return partition_collection


def polygon_collection_to_graph(polygon_collection, weights, original_split, is_contiguous, weight_attr="weight"):
    """ Convert collection of polygons to networkx graph

    Conversion of a polygon collection into a graph allows
    later graph partitioning
    :param polygon_collection:
    :param weights: weight of each polygon in collection
    :param original_split: "katana" or "hexana"
    :param is_contiguous: True or False (metis options)
    :param weight_attr: name of weight attribute
    :return:
    """
    if not is_iterable(polygon_collection):
        raise TypeError("Input must be a collection but is '{}'".format(type(polygon_collection)))

    if original_split == "katana":
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


def radius_of_curvature(line):
    """ Compute curvature radius of LineString

    :param line:
    :return:
    """
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

    divider = np.sqrt(np.fabs((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)))
    divider[divider == 0] = 0.1
    result = a * b * c / divider

    # Return values and add replicate to beginning of array (as result of curvature computation returns an array with
    #  length = length(line.coords) - 2): return array with length = length(line.coords) - 1
    return np.concatenate(([result[0]], result))


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
            new_collection.extend(method(geometry_collection, threshold))
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
