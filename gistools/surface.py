# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import numpy as np
from numpy import sin, pi, sqrt, abs

from gistools.coordinates import Ellipsoid
from gistools.utils.check.type import check_type


def compute_surface(lb_pixel, rb_pixel, ub_pixel, bb_pixel, geo_type: str, ellipsoid: Ellipsoid):
    """ Compute surface of earth pixels

    Combine surface element in spherical coordinates
    and local spherical approximation of the given
    ellipsoid to compute pixel surface. Or return
    surface from evenly spaced pixels from equiarea
    map projection.
    :param lb_pixel: left border pixel coordinate
    :param rb_pixel: right border pixel coordinate
    :param ub_pixel: upper border pixel coordinate
    :param bb_pixel: bottom border pixel coordinate
    :param geo_type: geo projection ("latlon" or "equal")
    :param ellipsoid: ellipsoid model
    :return:
    """

    latitude = (ub_pixel + bb_pixel) / 2

    def ellipsoid_normal():
        return ellipsoid.a / sqrt(1 - (ellipsoid.e ** 2) * (sin(latitude * pi / 180)) ** 2)

    def local_sphere_radius():

        p = ellipsoid.a * (1 - ellipsoid.e ** 2) / (1 - ellipsoid.e ** 2 *
                                                    sin(latitude * pi/180) ** 2) ** (3/2)
        return sqrt(ellipsoid_normal() * p) * 10 ** -3

    check_type(geo_type, str, ellipsoid, Ellipsoid)

    if geo_type == "latlon":
        surface = local_sphere_radius() ** 2 * (sin(ub_pixel * pi / 180) - sin(bb_pixel * pi / 180)) *\
                  (rb_pixel * pi / 180 - lb_pixel * pi / 180)
    elif geo_type == "equal":
        surface = abs(ub_pixel - bb_pixel) * abs(rb_pixel - lb_pixel)
    else:
        raise ValueError("Projection {} has not been defined yet".format(geo_type))

    return surface


def pixel_overlap_area(centroid1, centroid2, res1, res2, geoproj: str, ellipsoid: Ellipsoid):
    """ Retrieve overlay area between pixels

    Compute overlap area between pixels
    (thanks to https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap for inspiration)
    :param centroid1:
    :param centroid2:
    :param res1:
    :param res2:
    :param geoproj:
    :param ellipsoid:
    :return:
    """

    def get_pixel_borders(centroid, res):
        return centroid[0] - res/2, centroid[0] + res/2, centroid[1] + res/2, centroid[1] - res/2

    lb_pixel1, rb_pixel1, ub_pixel1, bb_pixel1 = get_pixel_borders(centroid1, res1)
    lb_pixel2, rb_pixel2, ub_pixel2, bb_pixel2 = get_pixel_borders(centroid2, res2)

    # Overlapping area
    lb_area = np.maximum(lb_pixel1, lb_pixel2)
    rb_area = np.minimum(rb_pixel1, rb_pixel2)
    ub_area = np.minimum(ub_pixel1, ub_pixel2)
    bb_area = np.maximum(bb_pixel1, bb_pixel2)

    # Overlapping (west-east and south-north)
    x_overlap = np.maximum(0, rb_area - lb_area)
    y_overlap = np.maximum(0, ub_area - bb_area)

    overlapping_area = compute_surface(lb_area, rb_area, ub_area, bb_area, geoproj, ellipsoid)
    overlapping_area[np.isclose(x_overlap, 0) | np.isclose(y_overlap, 0)] = 0

    return overlapping_area
