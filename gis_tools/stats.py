# -*- coding: utf-8 -*-

""" Statistic toolset for geographic objects

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
from math import sqrt as msqrt

import numpy as np

from gis_tools.layer import GeoLayer
from gis_tools.exceptions import ZonalStatisticsError
from gis_tools.raster import RasterMap
from utils.check import type_assert, check_type_in_collection
from utils.sys.timer import Timer

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


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
        self._layer_array = self._layer.to_array(raster_.geo_grid, "idx", data_type='uint32', all_touched=all_touched)

        # Raster
        self.raster = raster

    def mean(self):
        """ Compute zonal mean

        :return: list of mean values for each geometry zone
        """
        def average(raster):
            if self.is_surface_weighted:
                return [np.average(cell[~np.isnan(cell)], surf[~np.isnan(cell)]) for cell,
                        surf in self._get_raster_cell_values_with_surface(raster)]
            else:
                return [np.nanmean(cell) for cell in self._get_raster_cell_values(raster)]

        if isinstance(self.raster, list):
            return [average(r) for r in self.raster]
        else:
            return average(self.raster)

    def std(self):
        """ Compute zonal standard deviation

        :return: list of std values for each geometry zone
        """
        def standard_deviation(raster):
            if self.is_surface_weighted:
                return [weight_std(cell[~np.isnan(cell)], surf[~np.isnan(cell)]) for
                        cell, surf in self._get_raster_cell_values_with_surface(raster)]
            else:
                return [np.nanstd(cell) for cell in self._get_raster_cell_values(raster)]

        if isinstance(self.raster, list):
            return [standard_deviation(r) for r in self.raster]
        else:
            return standard_deviation(self.raster)

    ###################
    # Protected methods

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


def weight_std(values, weights):
    """ Return weighted standard deviation

    Use math.sqrt rather than numpy.sqrt for speed
    :param values:
    :param weights:
    :return:
    """

    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return msqrt(variance)


if __name__ == '__main__':
    from gis_tools.raster import DigitalElevationModel
    from gis_tools.layer import PolygonLayer
    dem = DigitalElevationModel("/home/benjamin/Documents/Post-doc Guyane/Data/DEM/srtm_guyana.tif",
                                no_data_value=-32768)
    # biomass = RasterMap("/home/benjamin/Documents/Post-doc Guyane/Data/Resource rasters/Biomasse "
    #                     "Guyane/AGB_map/biomasse_ressource.tif")
    # biomass = biomass.to_crs({'init': 'epsg:32622'})
    dem_utm = dem.to_crs({'init': 'epsg:32622'})
    slope = dem_utm.compute_slope()
    print(slope.raster_file)

    layer = PolygonLayer("/home/benjamin/Documents/Post-doc Guyane/Data/Geo layers/Parc amazonien/enp_pn_s_973.shp")
    zonal_stat = ZonalStatistics(slope, layer, is_surface_weighted=False, all_touched=True)
    with Timer() as t:
        avg = zonal_stat.mean()
    print("time: %s" % t)
    print(avg)
