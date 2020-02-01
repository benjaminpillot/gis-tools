# -*- coding: utf-8 -*-

""" Functions for computing get_horizon profiles

More detailed description.
"""

__version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


import numpy as np

from gistools.coordinates import Ellipsoid


# TODO: use Numba
# TODO: use Dozier algorithm (1981)
# TODO: use algorithm from A. James Stewart (1998)

def get_horizon(latitude, longitude, dem, ellipsoid=Ellipsoid("WGS84"), distance=0.5, precision=1):
    """ Compute local get_horizon obstruction from Digital Elevation Model

    This function is mainly based on a previous Matlab function
    (see https://fr.mathworks.com/matlabcentral/fileexchange/59421-dem-based-topography-get_horizon-model)
    :param latitude:
    :param longitude:
    :param dem: DigitalElevationModel instance
    :param ellipsoid: Ellipsoid instance
    :param distance: distance in degrees
    :param precision: azimuth precision of resulting horizon in degrees
    :return:
    """

    # Prune DEM and fit to study area
    study_area = dem.get_raster_at(ll_point=(latitude - distance, longitude - distance),
                                   ur_point=(latitude + distance, longitude + distance))

    y_obs, x_obs = study_area.geo_grid.latlon_to_2d_index(latitude, longitude)
    z_obs = study_area.get_value_at(latitude, longitude)

    # Azimuth and elevation
    azimuth = (180/np.pi) * get_azimuth(latitude * np.pi/180, longitude * np.pi/180,
                                        (study_area.geo_grid.latitude - dem.res/2) * np.pi/180,
                                        (study_area.geo_grid.longitude + dem.res/2) * np.pi/180, ellipsoid.e)
    elevation = np.zeros(azimuth.shape)
    elevation[study_area > z_obs] = \
        get_elevation(z_obs, study_area[study_area > z_obs], latitude * np.pi/180,
                      study_area.geo_grid.latitude[study_area > z_obs] * np.pi/180,
                      longitude * np.pi/180, study_area.geo_grid.longitude[study_area > z_obs]
                      * np.pi/180, ellipsoid.e, ellipsoid.a)
    # TODO: understand why "z_obs < study_area" return a numpy ValueError (ambiguous truth value)

    # Elevation vector length
    len_elevation = (90 + precision) // precision

    elevation_dic = dict(ne=np.zeros((y_obs, len_elevation)),
                         e=np.zeros((study_area.x_size - x_obs, 2*len_elevation - 1)),
                         s=np.zeros((study_area.y_size - y_obs, 2*len_elevation - 1)),
                         w=np.zeros((x_obs, 2*len_elevation - 1)), nw=np.zeros((y_obs, len_elevation)))

    azimuth_dic = dict(ne=np.arange(-180, -90 + precision, precision), e=np.arange(-180, 0 + precision, precision),
                       s=np.arange(-90, 90 + precision, precision), w=np.arange(0, 180 + precision, precision),
                       nw=np.arange(90, 180 + precision, precision))

    # Main computation
    # NE & NW
    for n, (az, el) in enumerate(zip(azimuth[:y_obs], elevation[:y_obs])):
        idx_ne = np.digitize(azimuth_dic["ne"], az[x_obs:])
        idx_nw = np.digitize(azimuth_dic['nw'], az[:x_obs])
        elevation_dic["ne"][n, idx_ne < len(az[x_obs:])] = el[x_obs:][idx_ne[idx_ne < len(az[x_obs:])]]
        elevation_dic["nw"][n, idx_nw < len(az[:x_obs])] = el[:x_obs][idx_nw[idx_nw < len(az[:x_obs])]]

    # South
    for n, (az, el) in enumerate(zip(azimuth[y_obs:, ::-1], elevation[y_obs:, ::-1])):
        idx_s = np.digitize(azimuth_dic["s"], az)
        elevation_dic["s"][n, idx_s < len(az)] = el[idx_s[idx_s < len(az)]]

    # East
    for n, (az, el) in enumerate(zip(azimuth[:, x_obs:].transpose(), elevation[:, x_obs:].transpose())):
        idx_e = np.digitize(azimuth_dic["e"], az)
        elevation_dic["e"][n, idx_e < len(az)] = el[idx_e[idx_e < len(az)]]

    # West
    for n, (az, el) in enumerate(zip(azimuth[::-1, :x_obs].transpose(), elevation[::-1, :x_obs].transpose())):
        idx_w = np.digitize(azimuth_dic["w"], az)
        elevation_dic["w"][n, idx_w < len(az)] = el[idx_w[idx_w < len(az)]]

    sun_mask = np.concatenate([elevation_dic[key].max(0) for key in elevation_dic.keys()])
    az_mask = np.concatenate([azimuth_dic[key] for key in azimuth_dic.keys()]) + 180

    horizon = dict(elevation=np.zeros((360 + precision)//precision),
                   azimuth=np.arange(0, 360 + precision, precision))
    for n, az in enumerate(horizon["azimuth"]):
        horizon["elevation"][n] = np.max(sun_mask[az_mask == az])

    horizon["elevation"][-1] = horizon["elevation"][0]

    return horizon


def geo_to_cartesian(latitude, longitude, altitude, e, a):
    """ Convert geographic to cartesian coordinates

    :param latitude:
    :param longitude:
    :param altitude:
    :param e: eccentricity
    :param a: semi-major axis
    :return:
    """
    n = a / np.sqrt(1 - (e**2) * (np.sin(latitude))**2)

    # return X, Y, Z
    return (n + altitude) * np.cos(longitude) * np.cos(latitude), (n + altitude) * np.sin(longitude) * np.cos(
        latitude), (n * (1 - e**2) + altitude) * np.sin(latitude)


def get_azimuth(lat1, lon1, lat2, lon2, e):
    """ Compute azimuth

    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :param e: eccentricity
    :return:
    """

    # Retrieve isometric latitude
    iso_lat1 = get_isometric_latitude(lat1, e)
    iso_lat2 = get_isometric_latitude(lat2, e)

    # Compute azimuth
    return np.arctan2((lon1 - lon2), (iso_lat1 - iso_lat2))


def get_elevation(h_a, h_b, latitude_a, latitude_b, longitude_a, longitude_b, e, a):
    """ Compute elevation angle

    :param h_a:
    :param h_b:
    :param latitude_a:
    :param latitude_b:
    :param longitude_a:
    :param longitude_b:
    :param e: eccentricity
    :param a: semi-major axis
    :return:
    """
    x_a, y_a, z_a = geo_to_cartesian(latitude_a, longitude_a, h_a, e, a)
    x_b, y_b, z_b = geo_to_cartesian(latitude_b, longitude_b, h_b, e, a)

    inner_prod = (x_b - x_a) * np.cos(longitude_a) * np.cos(latitude_a) + (y_b - y_a) * np.sin(longitude_a) * np.cos(
        latitude_a) + (z_b - z_a) * np.sin(latitude_a)

    # Cartesian norm
    norm = np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2 + (z_b - z_a) ** 2)

    return np.arcsin(inner_prod/norm) * 180/np.pi


def get_isometric_latitude(latitude, e):
    """ Calculate isometric latitude

    :param latitude:
    :param e: eccentricity
    :return:
    """
    term_1 = np.tan((np.pi/4) + (latitude/2))
    term_2 = ((1 - e * np.sin(latitude)) / (1 + e * np.sin(latitude))) ** (e / 2)

    return np.log(term_1 * term_2)
