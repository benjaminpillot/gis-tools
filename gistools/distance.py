# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import numpy as np
from numba import jit
from numpy import arcsin, cos, sin, pi, sqrt

from gistools.coordinates import Ellipsoid
from pyproj import Geod

# __all__ = ["great_circle", "euclidean_distance", "compute_distance", "pyproj_distance"]
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2017, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def great_circle(latitude_1, latitude_2, delta_longitude, ellipsoid: Ellipsoid):
    """ Compute geodetic distance

    Compute geographic distance between 2 points
    of an ellipsoid using great circle formula
    :param latitude_1:
    :param latitude_2:
    :param delta_longitude:
    :param ellipsoid: Ellipsoid class instance
    :return:
    """

    def ellipsoid_normal():
        return ellipsoid.a / sqrt(1 - (ellipsoid.e ** 2) * (sin(latitude_1 * pi / 180))
                                  ** 2)

    def local_sphere_radius():

        p = ellipsoid.a * (1 - ellipsoid.e ** 2) / (1 - ellipsoid.e ** 2 * sin(
                latitude_1 * pi/180) ** 2) ** (3/2)
        return sqrt(ellipsoid_normal() * p) * 10 ** -3

    # Haversine formula is numerically better conditioned for small distances
    # due to rounding errors on computer systems with low floating-point precision

    # cosine_law = arccos(cos(latitude_1 * pi/180) * cos(latitude_2 * pi/180) * cos(delta_longitude * pi/180) + sin(
    #     latitude_1 * pi/180) * sin(latitude_2 * pi/180))
    haversine_formula = 2 * arcsin(sqrt(sin((latitude_2 * pi/180 - latitude_1 * pi/180)/2) ** 2 +
                                        cos(latitude_1 * pi/180) * cos(latitude_2 * pi/180) * sin((delta_longitude *
                                                                                                   pi/180)/2) ** 2))

    return local_sphere_radius() * haversine_formula


@jit(nopython=True)
def euclidean_distance(x1, y1, x2, y2):
    """ Compute euclidean distance in a plane

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def pyproj_distance(x1, y1, x2, y2, ellipsoid_name):
    """ Compute geodesic using pyproj package

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param ellipsoid_name: name of ellipsoid
    :return: distance (in m)
    """
    geod = Geod(ellps=ellipsoid_name.upper())
    if np.size(x1) != np.size(x2):
        try:
            if np.size(x1) < np.size(x2):
                x1 = np.full(x2.shape, x1)
                y1 = np.full(x2.shape, y1)
            else:
                x2 = np.full(x1.shape, x2)
                y2 = np.full(x1.shape, y2)
        except ValueError:
            raise ValueError("x1/y1 and x2/y2 must have the same size or one pair must be scalar")

    return geod.inv(x1, y1, x2, y2)[2]


def compute_distance(x1, y1, x2, y2, geoproj: str, ellipsoid_model: Ellipsoid):

    # check_type(geoproj, str, ellipsoid_model, Ellipsoid)

    if geoproj == "latlon":
        # distance = great_circle(y1, y2, x1 - x2, ellipsoid_model)
        distance = pyproj_distance(x1, y1, x2, y2, ellipsoid_model.model)
    elif geoproj == "equal":
        distance = euclidean_distance(x1, y1, x2, y2)
    else:
        raise ValueError("Projection {} has not been defined yet".format(geoproj))

    return distance
