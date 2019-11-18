# -*- coding: utf-8 -*-

""" OpenStreetMap related tools

More detailed description.
"""

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


if __name__ == "__main__":
    import geopandas as gpd
    import osmnx as ox
    # test = ox.graph_from_place('Piedmont, California, USA', network_type='walk')
    place = 'Sao Sebastiao, Distrito Federal, Brasil'
    test = ox.gdf_from_place(place)
    polygon_coord_strs = ox.get_polygons_coordinates(test.geometry[0])
    response_jsons = []
    for poly_coord_str in polygon_coord_strs:
        query_template = '[out:json][timeout:180];(relation["place"]["place"="city_block"](poly:"{polygon}");>;);out;'
        query_str = query_template.format(polygon=poly_coord_str)
        response_jsons.append(ox.overpass_request(data={'data': query_str}, timeout=180))

    print(response_jsons)
    gdf = gpd.GeoDataFrame(response_jsons[0])
