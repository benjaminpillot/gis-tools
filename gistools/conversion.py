# -*- coding: utf-8 -*-

""" Geo conversion tools

Convert from/to shapefile/vector/geometry
from/to raster/array
"""

import os
import warnings
from tempfile import NamedTemporaryFile

import numpy as np
import pyproj
import rasterio
from rasterio import features
from rasterio.transform import Affine
from osgeo import gdal, ogr, osr

from gistools.coordinates import GeoGrid


# Allow OGR and GDAL to throw Python exceptions
from gistools.utils.check.type import check_type, type_assert

ogr.UseExceptions()
gdal.UseExceptions()


def slope_to_layer(dem, threshold, min_connection=1, simplify_tolerance=0, is_8_connected=False):
    """ Build polygon layer using slope classification


    :return:
    """
    from gistools.raster import RasterMap
    slope = dem.compute_slope()
    classification = RasterMap(np.zeros((slope.y_size, slope.x_size)), slope.geo_grid,
                               crs=slope.crs, no_data_value=0)
    classification[slope <= threshold] = 1
    classification_sieved = classification.apply_filter("sieve", min_connection)
    layer = classification_sieved.polygonize("slope", is_8_connected)
    layer = layer.simplify(simplify_tolerance)

    return layer


def shape_to_raster(shapefile: str, raster_file: str, geo_grid: GeoGrid,
                    geo_crs: str = "WGS84", burn_value=1, no_data_value=0):
    """ Convert shape to raster

    Convert input file (e.g shapefile)
    geometry into raster
    (see https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html)

    :param shapefile: path to shapefile to convert
    :param raster_file: path to raster file where to write
    :param geo_grid: geo grid which raster must rely on
    :param geo_crs: geographic coordinate system (see http://spatialreference.org/ref/epsg/wgs-84/
    for the WGS84 CRS for instance)
    :param burn_value: value to be burned from shape into raster
    :param no_data_value: value of raster cells where there are no data
    """
    check_type(shapefile, str, raster_file, str, geo_grid, GeoGrid, geo_crs, str)
    source_ds = ogr.Open(shapefile)
    if source_ds is None:
        raise RuntimeError("Unable to open {}".format(shapefile))

    # Retrieve bounding box of geometry
    source_layer = source_ds.GetLayer()

    # Create the destination data source
    driver = gdal.GetDriverByName('GTiff')

    # Be very careful here with geo_grid.num_x
    # and geo_grid.num_y : they are not the number
    # of pixels but the number of edges regarding
    # raster computation...
    target_ds = driver.Create(raster_file, geo_grid.num_x, geo_grid.num_y, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_grid.geo_transform)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)

    # Set geographic coordinate system
    projection = osr.SpatialReference()
    projection.SetWellKnownGeogCS(geo_crs)
    target_ds.SetProjection(projection.ExportToWkt())

    # Rasterization
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[burn_value])

    return 0


def shape_to_array(shapefile: str, geo_grid: GeoGrid, geo_crs: str = "WGS84"):
    """ Convert shape to numpy array

    Write shape to array using 1s
    for geometry and 0s for voids (no data)
    :param shapefile: path to shapefile that must be converted
    :param geo_grid: geo grid on which shape must be projected
    :param geo_crs: geographic coordinate system
    :return: numpy array and geo_grid

    :Example:
    >>> array, geo_grid = shape_to_array("/path/to/shape/file.shp", GeoGrid.from_geo_file(\
    "/path/to/shape/file.shp"))
    """
    # Make temp file for storing raster before writing to array
    raster_temp_file = NamedTemporaryFile(suffix='.tif')

    # Write shape to raster
    shape_to_raster(shapefile, raster_temp_file.name, geo_grid, geo_crs)

    # Convert raster to array
    return raster_to_array(raster_temp_file.name)


def geopandas_to_raster(gpd_frame, raster_file, gpd_column,
                        geo_grid: GeoGrid, datatype, all_touched):
    """ Write geopandas data frame to raster

    :param gpd_frame:
    :param raster_file:
    :param gpd_column:
    :param geo_grid: GeoGrid instance
    :param datatype: data type ('uint8', 'float32', etc.)
    :param all_touched:
    :return:
    """

    # _ = _create_raster_file(raster_file, geo_grid, geo_crs, 0)

    # Open temp raster file with rasterio and copy metadata
    # raster_fcn = rasterio.open(raster_temp_file.name)
    # meta = raster_fcn.meta.copy()
    # meta.update(compress='lzw')

    with rasterio.open(raster_file, 'w', driver="GTiff", width=geo_grid.num_x,
                       height=geo_grid.num_y, count=1, dtype=datatype, crs=gpd_frame.crs,
                       transform=Affine.from_gdal(*geo_grid.geo_transform)) as out:

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom, value) for geom, value in zip(gpd_frame.geometry, gpd_frame[gpd_column]))

        burned = features.rasterize(shapes=shapes, fill=0,
                                    out_shape=(geo_grid.num_y, geo_grid.num_x),
                                    dtype=datatype,
                                    transform=Affine.from_gdal(*geo_grid.geo_transform),
                                    all_touched=all_touched)
        out.write_band(1, burned)

    return 0


def geopandas_to_array(gpd_frame, gpd_column, geo_grid: GeoGrid, datatype, all_touched):
    """ Convert geopandas data frame to numpy array

    :param gpd_frame:
    :param gpd_column:
    :param geo_grid:
    :param datatype:
    :param all_touched:
    :return:
    """

    # Make temp file for storing raster before writing to array
    raster_temp_file = NamedTemporaryFile(suffix='.tif')

    # Write geopandas to raster
    geopandas_to_raster(gpd_frame, raster_temp_file.name,
                        gpd_column, geo_grid, datatype, all_touched)

    # Convert raster to array
    return raster_to_array(raster_temp_file.name)


@type_assert(raster_file=str, array=np.ndarray, geo_grid=GeoGrid)
def array_to_raster(raster_file: str, array: np.ndarray, geo_grid: GeoGrid,
                    crs, datatype=gdal.GDT_Byte, no_data_value=None):
    """ Convert numpy array to raster file

    Write numpy array to raster file
    :param raster_file: path to raster file
    :param array: numpy array
    :param geo_grid: GeoGrid class instance
    :param crs:
    :param datatype:
    :param no_data_value:
    :return:
    :Example:
    >>> array_to_raster("/path/to/my/raster.tif", np.array([[5,8,3], [6,7,8]]), GeoGrid())
    """
    try:
        crs = pyproj.CRS(crs)
    except pyproj.exceptions.CRSError:
        raise RuntimeError("Invalid CRS")

    if os.path.isfile(raster_file):
        warnings.warn("File {} already exists. Overwriting...".format(raster_file), Warning)

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(raster_file, array.shape[1], array.shape[0], 1, datatype)
    out_raster.SetGeoTransform(geo_grid.geo_transform)
    out_band = out_raster.GetRasterBand(1)
    if no_data_value is not None:
        out_band.SetNoDataValue(no_data_value)

    # Write array to raster band
    out_band.WriteArray(array)

    # Define spatial reference system
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromWkt(crs.to_wkt())
    # out_raster_srs.ImportFromProj4(crs)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())

    # Finalization : erase data from cache
    out_band.FlushCache()

    # Close dataset
    del out_raster

    # For matlab users not used to this syntax...
    # Be sure the function has just worked fine
    # by returning 0
    return 0


def raster_to_array(raster_file: str, band=None):
    """ Get numpy array from raster file

    :param raster_file: path to raster file
    :param band: raster band number
    :return: numpy array

    :Example:
    >>> array = raster_to_array("/path/to/my/raster.tif")
    """
    check_type(raster_file, str)

    source_ds = gdal.Open(raster_file)
    if band:
        source_ds = source_ds.GetRasterBand(band)

    return source_ds.ReadAsArray()
