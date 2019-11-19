# -*- coding: utf-8 -*-

""" OpenStreetMap related tools

More detailed description.
"""
import geojson
import requests
from osmxtract import overpass
from osmxtract.overpass import ql_query
from shapely.geometry import shape

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


DEFAULT_CRS = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'


def isvalid(geom):
    try:
        shape(geom)
        return 1
    except ValueError:
        return 0


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import geopandas as gpd
    import osmnx as ox
    # test = ox.graph_from_place('Piedmont, California, USA', network_type='walk')
    place = 'Sao Sebastiao, Distrito Federal, Brasil'
    test = ox.gdf_from_place(place)
    polygon_coord_strs = ox.get_polygons_coordinates(test.geometry[0])
    response_jsons = []
    for poly_coord_str in polygon_coord_strs:
        # query = ql_query(test.geometry[0].bounds, "place", values=["city_block"])
        query_template = '[out:json][timeout:180];nwr[place="city_block"](poly:"{polygon}");out geom qt;'
        query = query_template.format(polygon=poly_coord_str)
        # query = '[out:json][timeout:180];area[name="London"][admin_level="6"][boundary="administrative"]->.' \
        #         'londonarea; rel(pivot.londonarea);out geom;'
        # response = overpass.request(query)
        response = ox.overpass_request(data={'data': query}, timeout=180)
        # response2 = requests.get('http://overpass-api.de/api/interpreter', params={'data': query})

        feature_collection = overpass.as_geojson(response, 'multipolygons')
        gdf = gpd.GeoDataFrame.from_features(feature_collection)

    gdf.plot()
    plt.show()
