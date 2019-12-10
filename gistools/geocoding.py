# -*- coding: utf-8 -*-

""" All tools related to geocoding (Convert physical address in to coordinates)

More detailed description.
"""
from geopandas import GeoDataFrame
from itertools import combinations

from gistools.layer import PolygonLayer, cascaded_intersection
from utils.check import is_iterable

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def all_possible_addresses(list_of_polygon_layers, address_attribute_name='name', keeping_attributes=None):
    """ Return all possible addresses with uncertainty from multiple polygon layers

    :param list_of_polygon_layers: list of PolygonLayer instances corresponding to address zones (block, quarter,
    street, etc.)
    :param address_attribute_name: attribute name corresponding to address name in each layer
    :param keeping_attributes: optional attributes to keep in layers (list of str)
    :return:
    """
    if not is_iterable(list_of_polygon_layers):
        raise TypeError("Input must be an iterable but is '%s'" % type(list_of_polygon_layers).__name__)

    for layer in list_of_polygon_layers:
        if not isinstance(layer, PolygonLayer):
            raise TypeError("All layers in iterable must be of type 'PolygonLayer'")

    intersection_level = 0
    address_levels = ["level%d" % n for n in range(len(list_of_polygon_layers))]
    layers = [layer.rename(address_attribute_name, "level%d" % n) for n, layer in enumerate(list_of_polygon_layers)]
    addresses = []

    while "there are deeper levels of intersection":

        for comb in combinations(layers, intersection_level + 1):
            layer = cascaded_intersection(comb)
            layer['address'] = ",".join([level if level in layer.attributes() else "" for level in address_levels])
            layer['min_delta'], layer['max_delta'] = layer.distance_of_centroid_to_boundary()
            addresses.append(layer)

        if intersection_level == len(layers) - 1:
            break
        else:
            intersection_level += 1

    if keeping_attributes:
        keeping_attributes = ['address', 'min_delta', 'max_delta'].extend(keeping_attributes)
    else:
        keeping_attributes = ['address', 'min_delta', 'max_delta']

    addresses = [address.keep_attributes(keeping_attributes) for address in addresses]
    result = addresses[0]

    for address in addresses[1::]:
        result = result.append(address)

    return result


class Address:
    """ Address base super class

    """

    def __init__(self):
        pass


if __name__ == "__main__":
    admin_l10 = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_10.shp")
    admin_l11 = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_11.shp")
    admin_l10_l11 = admin_l10.overlay(admin_l11, how="union").explode().to_crs(epsg=32723)
    place_quarter = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_quarter.shp"
                                 "").to_crs(epsg=32723)
    city_block = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_city_block.shp"
                              "").to_crs(epsg=32723)

    test = all_possible_addresses([admin_l10_l11, place_quarter, city_block])
    test.to_file("test.shp")

    print(test)
