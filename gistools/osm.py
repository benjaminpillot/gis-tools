# -*- coding: utf-8 -*-

""" OpenStreetMap related tools

Among available tools:
- Use Overpass API to query OSM database and convert JSON
response to geopandas dataframe (thanks to https://github.com/yannforget/OSMxtract for inspiration !)
"""
import geojson
import geopandas as gpd
from osmnx import geocode_to_gdf
from osmnx.downloader import overpass_request

from gistools.exceptions import QlQueryError
from gistools.geometry import merge
from osmnx.settings import default_crs
from shapely.geometry import LineString, Point, MultiLineString, Polygon, MultiPolygon, MultiPoint

from gistools.utils.check.value import check_string

GEOMETRY_CLASS = {'linestring': (LineString, MultiLineString), 'polygon': (Polygon, MultiPolygon),
                  'point': (Point, MultiPoint)}


def _to_point_features(json):
    """ Read json response and extract point geometries

    :param json: JSON response from overpass API
    :return: GeoJSON FeatureCollection
    """
    features = []
    elements = [e for e in json['elements'] if e['type'] == 'node']
    for elem in elements:
        coords = [elem['lon'], elem['lat']]
        features.append(geojson.Feature(id=elem['id'], geometry=Point(coords), properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _to_features(json, geometry_type):
    """ Read json response and extract (multi)linestring/polygon geometries

    :param json: json response
    :param geometry_type: {'linestring', 'polygon'}
    :return:
    """
    features = []

    if geometry_type == 'linestring':
        elements = [e for e in json['elements'] if e['type'] == 'way' or e['type'] == 'relation']
    else:
        elements = [e for e in json['elements'] if e['type'] == 'way' or (e['type'] == 'relation' and e['tags']['type']
                                                                          in ('multipolygon', 'boundary'))]

    for elem in elements:

        if elem['type'] == 'way':
            coords = [[node['lon'], node['lat']] for node in elem['geometry']]
            try:
                geom = GEOMETRY_CLASS[geometry_type][0](coords)  # LineString, Polygon
            except ValueError:
                pass
            else:
                features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=_feature_tags(elem)))

        elif elem['type'] == 'relation':
            collection = []
            for member in elem['members']:
                if member['type'] == 'way':
                    member_coords = [(node['lon'], node['lat']) for node in member['geometry']]
                    collection.append(LineString(member_coords))
            geom_collection = merge(collection)

            if geom_collection:
                try:
                    geom = GEOMETRY_CLASS[geometry_type][1]([GEOMETRY_CLASS[geometry_type][0](line) for line in
                                                             geom_collection])  # MultiLineString, MultiPolygon
                except ValueError:
                    pass
                else:
                    features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _feature_tags(json_element):
    """ Update feature tags to set OSM ID and type

    :param json_element: 'elements' feature in JSON dict response
    :return:
    """
    if 'id' not in json_element['tags'].keys():
        tags = dict(osm_id=json_element['id'], **json_element['tags'])
    else:
        tags = json_element['tags']

    # Add osm type to attributes
    tags.update(osm_type=json_element['type'])

    return tags


def download_osm_features(place, osm_type, tag, values=None, by_poly=True, timeout=180):
    """ Download OSM features within given place

    :param place: single place name query (e.g: "London", "Bonn", etc.)
    :param osm_type: OSM geometry type str ('node', 'way', 'relation')
    :param tag: OSM tag to query
    :param values: str/list of possible values for the provided OSM tag
    :param by_poly: if True, retrieve features within polygon's list of coordinates, otherwise use bounds
    :param timeout:
    :return:
    """
    gdf_geometry = geocode_to_gdf(place)

    try:
        geometry = gdf_geometry.geometry[0]
    except AttributeError:  # Empty GeoDataFrame
        return None

    responses = []

    if by_poly:
        polygon_coord_strs = get_polygons_coordinates(geometry)
        for poly_coord_str in polygon_coord_strs:
            query = ql_query(osm_type, tag, values, polygon_coord=poly_coord_str, timeout=timeout)
            responses.append(overpass_request(data={'data': query}))
    else:
        query = ql_query(osm_type, tag, values, bounds=geometry.bounds, timeout=timeout)
        responses.append(overpass_request(data={'data': query}))

    return responses


def get_polygons_coordinates(geometry):
    """
    Extract exterior coordinates from polygon(s) to pass to OSM in a query by
    polygon. Ignore the interior ("holes") coordinates.

    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
        the geometry to extract exterior coordinates from

    Returns
    -------
    polygon_coord_strs : list

    Note
    ----
    Function from osmnx package version 0.10 (https://github.com/gboeing/osmnx)
    """

    # extract the exterior coordinates of the geometry to pass to the API later
    polygons_coords = []
    if isinstance(geometry, Polygon):
        x, y = geometry.exterior.xy
        polygons_coords.append(list(zip(x, y)))
    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry:
            x, y = polygon.exterior.xy
            polygons_coords.append(list(zip(x, y)))
    else:
        raise TypeError('Geometry must be a shapely Polygon or MultiPolygon')

    # convert the exterior coordinates of the polygon(s) to the string format
    # the API expects
    polygon_coord_strs = []
    for coords in polygons_coords:
        s = ''
        separator = ' '
        for coord in list(coords):
            # round floating point lats and longs to 6 decimal places (ie, ~100 mm),
            # so we can hash and cache strings consistently
            s = '{}{}{:.6f}{}{:.6f}'.format(s, separator, coord[1], separator, coord[0])
        polygon_coord_strs.append(s.strip(separator))

    return polygon_coord_strs


def json_to_geodataframe(response, geometry_type):
    """ Convert JSON responses to

    :param response: json response
    :param geometry_type: type of geometry to extract ('point', 'linestring', 'polygon', 'multipolygon')
    :return:
    """
    geometry_type = check_string(geometry_type, ('point', 'linestring', 'polygon'))

    if geometry_type == 'point':
        return gpd.GeoDataFrame.from_features(_to_point_features(response), crs=default_crs)
    else:
        return gpd.GeoDataFrame.from_features(_to_features(response, geometry_type), crs=default_crs)


def ql_query(osm_type, tag, values=None, bounds=None, polygon_coord=None, timeout=180):
    """ QL query (thanks to https://github.com/yannforget/OSMxtract for inspiration !)

    :param osm_type: OSM geometry type str {'node', 'way', 'relation', 'nwr'}
    :param tag: OSM tag to query
    :param values: str/list of possible values for the provided OSM tag
    :param bounds: geometry bounds
    :param polygon_coord: location's polygon list of coordinates
    :param timeout:
    :return:
    """
    osm_type = check_string(osm_type, ('node', 'way', 'relation', 'nwr'))

    if isinstance(values, str):
        values = [values]

    if bounds and not polygon_coord:
        west, south, east, north = bounds
        boundary = f'({south:.6f},{west:.6f},{north:.6f},{east:.6f})'
    elif polygon_coord and not bounds:
        boundary = f'(poly:"{polygon_coord}")'
    else:
        raise QlQueryError("Must define either geometry bounds or polygon coordinates")

    if values:
        if len(values) > 1:
            tags = f'["{ tag }"~"{ "|".join(values) }"]'
        else:
            tags = f'["{ tag }"="{ values[0] }"]'
    else:
        tags = f'["{tag}"]'

    return f'[out:json][timeout:{timeout}];{osm_type}{tags}{boundary};out geom;'
