# -*- coding: utf-8 -*-

""" OpenStreetMap related tools

More detailed description.
"""
import geojson
from gistools.exceptions import QlQueryError
from gistools.geometry import merge
from osmnx import gdf_from_place, get_polygons_coordinates, overpass_request
from osmnx.settings import default_crs
from shapely.geometry import LineString, MultiPolygon, Polygon
from utils.check import check_string

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def download_osm_features(place, osm_type, tag, values=None, by_poly=True, timeout=180):
    """ Download OSM features within given place

    :param place: location geocoding string
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


def extract_multipolygon_features(json):
    """ Read json response and extract multipolygon geometries

    :param json: json response from overpass API
    :return: GeoJSON FeatureCollection
    """
    features = []
    elements = [e for e in json['elements'] if e.get('type') == 'relation' and e['tags']['type'] == 'multipolygon']
    for elem in elements:
        collection = []
        for member in elem['members']:
            member_coords = []
            for node in member['geometry']:
                member_coords.append([node['lon'], node['lat']])
            collection.append(LineString(member_coords))
        geom_collection = merge(collection)

        if len(geom_collection) > 1:
            geom = MultiPolygon([Polygon(line) for line in geom_collection])
        else:
            geom = Polygon(geom_collection[0])

        if 'id' not in elem['tags'].keys():
            tags = dict(osm_id=elem['id'], **elem['tags'])
        else:
            tags = elem['tags']

        # Add osm type to attributes
        tags.update(osm_type=elem['type'])

        features.append(geojson.Feature(id=elem['id'], geometry=geom, properties=tags))

    return geojson.FeatureCollection(features)


def json_to_geodataframe(response, geometry_type):
    """ Convert JSON responses to

    :param response: json response
    :param geometry_type: type of geometry to extract ('point', 'linestring', 'polygon', 'multipolygon')
    :return:
    """
    geometry_type = check_string(geometry_type, ('point', 'linestring', 'polygon', 'multipolygon'))

    if geometry_type == 'point':
        pass
    elif geometry_type == 'linestring':
        pass
    elif geometry_type == 'polygon':
        pass
    elif geometry_type == 'multipolygon':
        return gpd.GeoDataFrame.from_features(extract_multipolygon_features(response), crs=default_crs)


def ql_query(osm_type, tag, values=None, bounds=None, polygon_coord=None, timeout=180):
    """ QL query (thantks to https://github.com/yannforget/OSMxtract for inspiration !)

    :param osm_type: OSM geometry type str {'node', 'way', 'relation'}
    :param tag: OSM tag to query
    :param values: str/list of possible values for the provided OSM tag
    :param bounds: geometry bounds
    :param polygon_coord: location's polygon list of coordinates
    :param timeout:
    :return:
    """
    osm_type = check_string(osm_type, ('node', 'way', 'relation'))

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
            query = f'["{ tag }"~"{ "|".join(values) }"]'
        else:
            query = f'["{ tag }"="{ values[0] }"]'
    else:
        query = f'["{tag}"]'

    return f'[out:json][timeout:{timeout}];{osm_type}{query}{boundary};out geom;'


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import geopandas as gpd
    # test = ox.graph_from_place('Piedmont, California, USA', network_type='walk')
    jsons = download_osm_features('Sao Sebastiao, Distrito Federal, Brasil', 'relation', 'place', 'city_block')
    gdf = json_to_geodataframe(jsons[0], 'multipolygon')
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
