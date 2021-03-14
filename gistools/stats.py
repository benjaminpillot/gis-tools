# -*- coding: utf-8 -*-

""" Statistic toolset for geographic objects

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
from math import sqrt as msqrt

import numpy as np

from gistools.layer import GeoLayer
from gistools.exceptions import ZonalStatisticsError
from gistools.raster import RasterMap
from gistools.utils.check.type import type_assert, check_type_in_collection


class ZonalStatistics:
    """ Zonal statistics class

    Toolset for computing statistics over a raster
    within geometries of a geo layer
    """

    @type_assert(raster=(RasterMap, list), layer=GeoLayer, is_surface_weighted=bool, all_touched=bool)
    def __init__(self, raster, layer, is_surface_weighted=False, all_touched=False):
        """ ZonalStatistics class constructor

        :param raster: RasterMap instance or list of RasterMap instances with the same geo-referencing
        :param layer: GeoLayer instance
        :param is_surface_weighted: Either weight stats by surface or not (boolean)
        """
        if isinstance(raster, list):
            check_type_in_collection(raster, RasterMap)
            if not RasterMap.is_equally_referenced(*raster):
                raise ZonalStatisticsError("All rasters must have the same geo-referencing")
            raster_ = raster[0]
        else:
            raster_ = raster

        if is_surface_weighted:
            self._surface = raster_.surface()

        self.is_surface_weighted = is_surface_weighted

        # Layer
        self._layer = layer.to_crs(raster_.crs)
        self._layer_index = self._layer.index + 1
        self._layer["idx"] = self._layer_index

        # Convert layer to array of indices
        self._layer_array = self._layer.to_array(raster_.geo_grid, "idx",
                                                 data_type='uint32', all_touched=all_touched)

        # Raster
        self.raster = raster

    def density(self, patch_value):
        """ Compute density of specified patch value

        Compute density (proportion of some value within zone).
        Better to be used with some raster classification
        :param patch_value:
        :return:
        """
        return self._get_statistic(density, weight_density, value=patch_value)

    # TODO: implement other statistical methods (min, max, count, etc.)
    def min(self):
        """ Compute zonal min

        :return:
        """
        return self._get_statistic(method=np.min)

    def max(self):
        """ Compute zonal max

        :return:
        """
        return self._get_statistic(method=np.max)

    def mean(self):
        """ Compute zonal mean

        :return: list of mean values for each geometry zone
        """
        return self._get_statistic(method=np.mean, weight_method=weight_average)

    def no_data_count(self, normalized: bool = True):
        """ Count no data (NaN) within zone

        :return:
        """
        def count(raster):
            if normalized:
                return [cell[np.isnan(cell)].size / cell.size if cell.size != 0 else np.nan for cell in
                        self._get_raster_cell_values(raster)]
            else:
                return [cell[np.isnan(cell)].size for cell in self._get_raster_cell_values(raster)]

        if isinstance(self.raster, list):
            return [count(r) for r in self.raster]
        else:
            return count(self.raster)

    def std(self):
        """ Compute zonal standard deviation

        :return: list of std values for each geometry zone
        """
        return self._get_statistic(method=np.std, weight_method=weight_std)

    def sum(self):
        """ Compute zonal sum

        :return: list of sum values for each geometry zone
        """
        return self._get_statistic(method=np.sum)

    ###################
    # Protected methods

    def _get_statistic(self, method, weight_method=None, **kwargs):
        def statistic(raster):
            if self.is_surface_weighted and weight_method is not None:
                return [weight_method(cell[~np.isnan(cell)], weights=surf[~np.isnan(cell)], **kwargs) for cell, surf in
                        self._get_raster_cell_values_with_surface(raster)]
            else:
                return [method(cell[~np.isnan(cell)], **kwargs)
                        for cell in self._get_raster_cell_values(raster)]

        if isinstance(self.raster, list):
            return [statistic(r) for r in self.raster]
        else:
            return statistic(self.raster)

    def _get_raster_cell_values(self, raster):
        n = 0
        while n < len(self._layer):
            yield raster[self._layer_array == self._layer_index[n]]
            n += 1

    def _get_raster_cell_values_with_surface(self, raster):
        n = 0
        while n < len(self._layer):
            yield raster[self._layer_array == self._layer_index[n]], self._surface[self._layer_array ==
                                                                                   self._layer_index[n]]
            n += 1


def weight_average(values, weights):
    """ Return weighted average

    Return weighted average, and None if sum(w) = 0
    :param values:
    :param weights:
    :return:
    """
    try:
        return np.average(values, weights=weights)
    except ZeroDivisionError:
        return None


def weight_std(values, weights):
    """ Return weighted standard deviation

    Use math.sqrt rather than numpy.sqrt for speed
    :param values:
    :param weights:
    :return:
    """
    try:
        average = np.average(values, weights=weights)
    except ZeroDivisionError:
        return None
    variance = np.average((values - average)**2, weights=weights)
    return msqrt(variance)


def weight_density(values, weights, value=None):
    """ Return weighted density

    :param values:
    :param weights:
    :param value:
    :return:
    """
    return weights[values == value].sum() / weights.sum()


def density(values, value):
    """ Return density of value among values

    :param values:
    :param value:
    :return: density (between 0 and 1)
    """

    return values[values == value].size / values.size
