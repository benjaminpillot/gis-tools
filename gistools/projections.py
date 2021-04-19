# -*- coding: utf-8 -*-

""" Tools related to geographic projections

More detailed description.
"""

# import

# __all__ = []
# __version__ = '0.1'
import pyproj

from osgeo import gdal, osr, ogr

from gistools.utils.check.type import type_assert
from gistools.utils.check.value import check_file

osr.UseExceptions()


def proj4_from_raster(raster_file: str):
    """ Get pyproj Proj4 projection from raster file

    :param raster_file:
    :return:
    """
    check_file(raster_file)

    try:
        proj = gdal.Open(raster_file).GetProjection()
    except RuntimeError:
        raise ValueError("Unable to read raster file '{}'".format(raster_file))

    return proj4_from_wkt(proj)


def proj4_from_layer(layer_file: str):
    """ Get pyproj Proj4 projection from layer file

    :param layer_file:
    :return:
    """
    check_file(layer_file)

    try:
        src_ds = ogr.Open(layer_file)
        return src_ds.GetLayer().GetSpatialRef().ExportToProj4()
    except RuntimeError:
        raise ValueError("Unable to read layer file '%s'" % layer_file)


def proj4_from_wkt(wkt):
    """ Convert wkt srs to proj4

    :param wkt:
    :return:
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)

    return srs.ExportToProj4()


def proj4_from(proj):
    """ Convert projection to proj4 string

    Convert projection string, dictionary, etc.
    to proj4 string
    :param proj:
    :return:
    """
    if type(proj) == int:
        try:
            proj4_str = pyproj.Proj('epsg:%d' % proj).srs
        except (ValueError, RuntimeError):
            raise ValueError("Invalid EPSG code")
    elif type(proj) == str or type(proj) == dict:
        try:
            proj4_str = pyproj.Proj(proj).srs
        except RuntimeError:
            try:
                proj4_str = proj4_from_wkt(proj)
            except (RuntimeError, TypeError):
                raise ValueError("Invalid projection string or dictionary")
    elif type(proj) == pyproj.Proj:
        proj4_str = proj.srs
    else:
        raise ValueError("Invalid projection format: '{}'".format(type(proj)))

    return proj4_str


@type_assert(proj1=(int, str, dict, pyproj.Proj), proj2=(int, str, dict, pyproj.Proj))
def is_equal(proj1, proj2):
    """ Compare 2 projections

    :param proj1:
    :param proj2:
    :return: True or False
    """
    # From an idea from https://github.com/jswhit/pyproj/issues/15
    # Use OGR library to compare projections
    srs = [srs_from(proj1), srs_from(proj2)]

    boolean = (False, True)
    return boolean[srs[0].IsSame(srs[1])]


def ellipsoid_from(proj):
    """ Get Ellipsoid model from projection

    :param proj:
    :return:
    """
    from gistools.coordinates import Ellipsoid

    srs = srs_from(proj)
    ellps = srs.GetAttrValue('Spheroid').replace(' ', '')  # Be sure all spaces in string are removed

    return Ellipsoid(ellps)


def srs_from(proj):
    """ Get spatial reference system from projection

    :param proj: str
        'epsg:code' or 'proj=proj_name'
    :return: SpatialReference instance (osgeo.osr package)
    """
    crs = pyproj.CRS(proj)
    # proj4 = proj4_from(proj)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs.to_wkt())

    return srs


def wkt_from(proj):
    """ Get WKT spatial reference system from projection

    :param proj:
    :return:
    """
    return srs_from(proj).ExportToWkt()


def crs_from_layer(layer):
    """ Get pyproj CRS from layer file

    Parameters
    ----------
    layer: str
        file name

    Returns
    -------

    """
    ds = ogr.Open(layer)
    crs = pyproj.CRS(ds.GetLayer().GetSpatialRef().ExportToWkt())
    ds = None

    return crs


def crs_from_raster(raster):
    """ Get pyproj CRS from raster file

    Parameters
    ----------
    raster

    Returns
    -------

    """
    ds = gdal.Open(raster)
    crs = pyproj.CRS(ds.GetProjection())
    ds = None

    return crs
