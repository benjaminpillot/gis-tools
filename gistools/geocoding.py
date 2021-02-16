# -*- coding: utf-8 -*-

""" All tools related to geocoding (Convert physical address in to coordinates)

More detailed description.
"""
from abc import abstractmethod
from collections import OrderedDict
from itertools import combinations

from numpy import unique

from gistools.exceptions import DictionaryConverterError
from gistools.layer import PolygonLayer, cascaded_intersection, concat_layers, LineLayer
from pandas import read_csv

from gistools.utils.check.descriptor import protected_property
from gistools.utils.check.type import is_iterable


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
    result[to] = result["level0"].str.cat([result[level] for level in address_levels[1::]], sep=',', na_rep="")
    result = result.dissolve(by=to)
    result['min_delta'], result['max_delta'] = result.distance_of_centroid_to_boundary()

    return result


class Address:
    """ Address base super class

    """
    addresses = protected_property("addresses")
    place = protected_property("place")
    layers = protected_property("layers")

    def __init__(self, place):
        """ Build class instance

        :param place:
        """
        self._place = place
        self._layers = None
        self._addresses = None

    def all_addresses(self, street_buffer=20):
        """ Retrieve all addresses corresponding to layers

        :param street_buffer:
        :return:
        """
        list_of_layers = [layer.buffer(street_buffer) if layer.name == "highway"
                          else layer for layer in self._layers]
        self._addresses = all_addresses(list_of_layers, by='name', to='address')

        return self

    def get_osm_layers(self, tags, crs=None, **kwargs):
        """ Retrieve OSM layers used for addressing (admin levels, streets, etc.)

        :param tags: list of tag/values tuples (e.g.: [("admin_level", ("10","11")), ("highway")]
        :param crs: set either crs or epsg code
        :param kwargs: keyword arguments of "from_osm" GeoLayer method
        :return:
        """
        self._layers = []
        for tag in tags:

            if len(tag) == 1:
                key, val = tag[0], None
            else:
                key, val = tag[0], tag[1]

            if key == "highway":
                layer = LineLayer.from_osm(self._place, key, val, **kwargs)
            else:
                layer = PolygonLayer.from_osm(self._place, key, val, **kwargs)

            self._layers.append(layer.to_crs(crs=crs))

        return self

    def geocode(self, address_converter):
        """ Geocode addresses using converter model

        :param address_converter: AddressParser instance
        :return:
        """
        pass


class AddressParser:
    """ Parse address using corresponding model

    """

    def __init__(self):
        pass

    @abstractmethod
    def parse(self, *args, **kwargs):
        pass


class XParser(AddressParser):
    """ XParser is an address parser gathering linguistic techniques
     for parsing addresses (regular expressions + edition distance)

    """
    _re_parser = OrderedDict()

    def __init__(self):
        """

        """
        super().__init__()

    def parse(self, *args, **kwargs):
        pass
