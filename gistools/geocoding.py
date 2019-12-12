# -*- coding: utf-8 -*-

""" All tools related to geocoding (Convert physical address in to coordinates)

More detailed description.
"""
from itertools import combinations

from gistools.layer import PolygonLayer, cascaded_intersection, concat_layers
from utils.check import is_iterable

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def all_addresses(list_of_polygon_layers, by='name', to='name', keeping_attributes=None):
    """ Return all possible addresses by intersecting multiple overlapping polygon layers

    This function is specifically designed for address geocoding. By intersecting
    all layers and combining the attribute corresponding to some specific address
    level ("name", such as street, city block, quarter, etc.), we retrieve a geo
    database with all possible geographical locations (polygon), corresponding
    address levels and one address string (e.g.: "Quarter X, street Y, block Z")
    :param list_of_polygon_layers: list of PolygonLayer instances
    :param by: attribute for which we must combine values (str) into final str description
    :param to: final attribute aggregating input layers' "by" attributes into one string
    :param keeping_attributes: optional attributes to keep in layers (list of str)
    :return:
    """
    if keeping_attributes is None:
        keeping_attributes = [by]
    else:
        keeping_attributes = [by] + keeping_attributes

    if not is_iterable(list_of_polygon_layers):
        raise TypeError("Input must be an iterable but is '%s'" % type(list_of_polygon_layers).__name__)

    for layer in list_of_polygon_layers:
        if not isinstance(layer, PolygonLayer):
            raise TypeError("All layers in iterable must be of type 'PolygonLayer'")

    addresses = []
    intersection_level = 0
    address_levels = ["level%d" % n for n in range(len(list_of_polygon_layers))]
    layers = [layer.keep_attributes(keeping_attributes).rename(by, "level%d" % n)
              for n, layer in enumerate(list_of_polygon_layers)]

    while "there are deeper levels of intersection":

        for comb in combinations(layers, intersection_level + 1):
            addresses.append(cascaded_intersection(comb))

        if intersection_level == len(layers) - 1:
            break
        else:
            intersection_level += 1

    result = concat_layers(addresses)
    result['min_delta'], result['max_delta'] = result.distance_of_centroid_to_boundary()
    result[to] = result["level0"].str.cat([result[level] for level in address_levels[1::]], sep=',', na_rep="")

    return result.dissolve(by=to)


class Address:
    """ Address base super class

    """

    def __init__(self):
        pass

    def all_addresses(self):
        """

        :return:
        """

    def geocode(self):
        pass


if __name__ == "__main__":
    from gistools.layer import LineLayer
    from utils.sys.timer import Timer
    admin_l10 = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_10.shp")
    admin_l11 = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_11.shp")
    admin_l10_l11 = admin_l10.overlay(admin_l11, how="union").explode().to_crs(epsg=32723)
    admin_l10_l11["name"] = admin_l10_l11["name_1"].str.cat(admin_l10_l11["name_2"], sep="", na_rep="")
    admin_l10_l11 = admin_l10.append(admin_l11).explode().to_crs(epsg=32723)
    zone = admin_l10_l11.drop_attribute(admin_l10_l11.attributes())
    place_quarter = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_quarter.shp"
                                 "").to_crs(epsg=32723)
    city_block = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_city_block.shp"
                              "").to_crs(epsg=32723)
    street = LineLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/highway.shp").to_crs(epsg=32723)
    street = street.overlay(zone, how="intersection")
    street = street.buffer(25).explode()

    with Timer() as t:
        test = all_addresses([admin_l10_l11, place_quarter, city_block, street], to='address')
    print("spent time: %s" % t)
    test.to_file("/home/benjamin/Desktop/APUREZA/geocoding/addresses.shp")
