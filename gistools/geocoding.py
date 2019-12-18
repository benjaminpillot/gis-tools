# -*- coding: utf-8 -*-

""" All tools related to geocoding (Convert physical address in to coordinates)

More detailed description.
"""
from abc import abstractmethod
from itertools import combinations

from numpy import unique

from gistools.exceptions import DictionaryConverterError
from gistools.layer import PolygonLayer, cascaded_intersection, concat_layers, LineLayer
from pandas import read_csv
from utils.check import is_iterable, protected_property, check_string

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
        list_of_layers = [layer.buffer(street_buffer) if layer.name == "highway" else layer for layer in self._layers]
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

        :param address_converter: AddressConverter instance
        :return:
        """
        pass


class AddressConverter:
    """ Convert address using corresponding model

    """

    def __init__(self):
        pass

    @abstractmethod
    def convert(self, *args, **kwargs):
        pass


class DictionaryConverter(AddressConverter):
    """

    """
    _dictionary = None
    _dictionary_columns = ('old', 'new', 'level')

    _address_levels = None

    def __init__(self, dictionary_file):
        """

        :param dictionary_file:
        """
        super().__init__()
        self.dictionary = dictionary_file
        self._address_levels = unique(self.dictionary["level"])

    def convert(self, addresses):
        """ Convert old address string to new format

        :param addresses: pandas Series
        :return:
        """
        converted_addresses = addresses.copy()
        for add_level in self._address_levels:
            level_dic = self.dictionary[self.dictionary["level"] == add_level]
            for n, address in enumerate(addresses):
                old_new = level_dic['old'].apply(lambda old: old in address) & \
                          level_dic['new'].apply(lambda new: new not in address)
                converter = level_dic[old_new].squeeze()

                if converter.size != 0:
                    converted_addresses[n] = converted_addresses[n].replace(converter['old'], converter['new'])

        return converted_addresses

        # new_address = address.apply(lambda adr: adr.replace(self.dictionary["old"], self.dictionary["new"]))

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, dictionary_file):
        dic = read_csv(dictionary_file, keep_default_na=False)
        for key in dic.columns:
            try:
                check_string(key, self._dictionary_columns)
            except ValueError:
                raise DictionaryConverterError("Dictionary must have the 3 following columns: ('old', 'new', 'level')")
        self._dictionary = dic


class LevenshteinConverter(AddressConverter):
    """

    """

    def convert(self):
        pass


if __name__ == "__main__":
    from utils.sys.timer import Timer
    import pandas as pd
    test = DictionaryConverter("/home/benjamin/Desktop/APUREZA/geocoding/dictionary/dictionary.csv")
    dengue = pd.read_csv("/home/benjamin/Desktop/APUREZA/geocoding/dengue_database/dengue.csv")
    addr = dengue["NM_LOGRADO"].str.cat([dengue["NM_COMPLEM"], dengue["NM_REFEREN"]], sep=',', na_rep="")
    with Timer() as t:
        new_addr = test.convert(addr)
    print("spent time: %s" % t)

    # from gistools.layer import LineLayer
    # test = Address("Sao Sebastiao, Distrito Federal")
    # test.get_osm_layers([("admin_level", ("10", "11")), ("place", "quarter"), ("place", "city_block"),
    #                      ("highway",)], crs=32723).all_addresses()
    # test.addresses.to_file("/home/benjamin/Desktop/APUREZA/geocoding/addresses.shp")


    # admin_l10_l11 = PolygonLayer.from_osm("Sao Sebastiao, Distrito Federal", 'admin_level', ("10", "11")).to_crs(
    #     epsg=32723)
    # place_quarter = PolygonLayer.from_osm()
    # zone = admin_l10_l11.drop_attribute(admin_l10_l11.attributes())
    # place_quarter = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_quarter.shp"
    #                              "").to_crs(epsg=32723)
    # city_block = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/place_city_block.shp"
    #                           "").to_crs(epsg=32723)
    # street = LineLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/highway.shp").to_crs(epsg=32723)
    # street = street.overlay(zone, how="intersection")
    # street = street.buffer(25).explode()
    #
    # with Timer() as t:
    #     test = all_addresses([admin_l10_l11, place_quarter, city_block, street], to='address')
    # print("spent time: %s" % t)
    # test.to_file("/home/benjamin/Desktop/APUREZA/geocoding/addresses.shp")
