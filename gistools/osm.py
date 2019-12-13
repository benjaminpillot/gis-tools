# -*- coding: utf-8 -*-

""" OpenStreetMap related tools

Among available tools:
- Use Overpass API to query OSM database and convert JSON
response to geopandas dataframe (thanks to https://github.com/yannforget/OSMxtract for inspiration !)
"""
import geojson
import geopandas as gpd
from gistools.exceptions import QlQueryError
from gistools.geometry import merge
from osmnx import gdf_from_place, get_polygons_coordinates, overpass_request
from osmnx.settings import default_crs
from pandas import concat
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
from utils.check import check_string

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def _to_linestring_features(json):
    """ Read JSON response and extract linestring features

    :param json: JSON response from overpass API
    :return:
    """
    features = []
    elements = [e for e in json['elements'] if e.get('type') == 'way']
    for elem in elements:
        coords = [[node['lon'], node['lat']] for node in elem['geometry']]
        geom = LineString(coords)
        features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _to_multipolygon_features(json):
    """ Read json response and extract multipolygon features (geometry + attributes)

    :param json: JSON response from overpass API
    :return: GeoJSON FeatureCollection
    """
    features = []
    # 12/13/2019: change to add multiple types in tags ("multipolygon" AND "boundary")
    elements = [e for e in json['elements'] if e.get('type') == 'relation' and e['tags']['type'] in ('multipolygon',
                                                                                                     'boundary')]
    for elem in elements:
        collection = []
        for member in elem['members']:
            if member['type'] == 'way':
                member_coords = [(node['lon'], node['lat']) for node in member['geometry']]
                collection.append(LineString(member_coords))
        geom_collection = merge(collection)

        if geom_collection:
            try:
                geom = MultiPolygon([Polygon(line) for line in geom_collection])
            except ValueError:
                pass
            else:
                features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _to_point_features(json):
    """ Read json response and extract point geometries

    :param json: JSON response from overpass API
    :return: GeoJSON FeatureCollection
    """
    features = []
    elements = [e for e in json['elements'] if e.get('type') == 'node']
    for elem in elements:
        coords = [elem['lon'], elem['lat']]
        features.append(geojson.Feature(id=elem['id'], geometry=Point(coords), properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _to_polygon_features(json):
    """ Read json response and extract polygon geometries

    :param json:
    :return:
    """
    features = []
    elements = [e for e in json['elements'] if e.get('type') == 'way']
    for elem in elements:
        coords = [[node['lon'], node['lat']] for node in elem['geometry']]
        try:
            geom = Polygon(coords)
        except ValueError:
            pass
        else:
            features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=_feature_tags(elem)))

    return geojson.FeatureCollection(features)


def _to_multi_line_features(json):
    pass


def _to_multi_point_features(json):
    """ Read json response and extract multi-node (relation) geometries

    Some OSM relations correspond to nodes.
    see https://wiki.openstreetmap.org/wiki/Types_of_relation
    :param json:
    :return:
    """
    pass


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
    gdf_geometry = gdf_from_place(place)

    try:
        geometry = gdf_geometry.geometry[0]
    except AttributeError:  # Empty GeoDataFrame
        return None

    responses = []

    if by_poly:
        polygon_coord_strs = get_polygons_coordinates(geometry)
        for poly_coord_str in polygon_coord_strs:
            query = ql_query(osm_type, tag, values, polygon_coord=poly_coord_str, timeout=timeout)
            responses.append(overpass_request(data={'data': query}, timeout=180))
    else:
        query = ql_query(osm_type, tag, values, bounds=geometry.bounds, timeout=timeout)
        responses.append(overpass_request(data={'data': query}, timeout=180))

    return responses


def json_to_geodataframe(response, geometry_type):
    """ Convert JSON responses to

    :param response: json response
    :param geometry_type: type of geometry to extract ('point', 'linestring', 'polygon', 'multipolygon')
    :return:
    """
    geometry_type = check_string(geometry_type, ('point', 'linestring', 'polygon'))

    if geometry_type == 'point':
        return gpd.GeoDataFrame.from_features(_to_point_features(response), crs=default_crs)
    elif geometry_type == 'linestring':
        return gpd.GeoDataFrame.from_features(_to_linestring_features(response), crs=default_crs)
    elif geometry_type == 'polygon':
        poly_df = gpd.GeoDataFrame.from_features(_to_polygon_features(response), crs=default_crs)
        multi_poly_df = gpd.GeoDataFrame.from_features(_to_multipolygon_features(response), crs=default_crs)

        return concat([poly_df, multi_poly_df], ignore_index=True, sort=False)


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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # test = ox.graph_from_place('Piedmont, California, USA', network_type='walk')
    jsons = download_osm_features('Sao Sebastiao, Distrito Federal, Brasil', 'way', 'building')
    gdf = json_to_geodataframe(jsons[0], 'linestring')
    gdf.to_file('test.shp')

    # test = ox.gdf_from_place(place)
    # polygon_coord_strs = ox.get_polygons_coordinates(test.geometry[0])
    # response_jsons = []
    # for poly_coord_str in polygon_coord_strs:
    #     # query = ql_query(test.geometry[0].bounds, "place", values=["city_block"])
    #     query_template = '[out:json][timeout:180]; rel["place"="city_block"](poly:"{polygon}");out geom;'
    #     query = query_template.format(polygon=poly_coord_str)
    #     # query = '[out:json][timeout:180];area[name="{}"]->.' \
    #     #         'londonarea; rel[place="city_block"](pivot.londonarea);out geom;'.format(place)
    #     # response = overpass.request(query)
    #     response = ox.overpass_request(data={'data': query}, timeout=180)
    #     # response2 = requests.get('http://overpass-api.de/api/interpreter', params={'data': query})
    #
    #     feature_collection = overpass.as_geojson(response, 'multipolygons')
    #     gdf = gpd.GeoDataFrame.from_features(feature_collection)

    gdf.plot()
    plt.show()
    print(gdf.columns)
