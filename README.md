# GisTools
Some geographical tools for python developers

## Introduction
GisTools is a small Python library for performing geographical computations. Typically, it gathers different tools from well-known libraries such as gdal, rasterio, geopandas, fiona and shapely.
It allows easily mixing operations between vectors and raster maps (multi-band raster are not supported at the moment).

## Basic operations
GisTools allows some of the following operations:
- [x] Fast polygon intersection and split
- [x] Polygon partition based on graph theory
- [x] Basic networking (shortest path)
- [x] Downloading DEM from online databases
- [x] Extract raster statistics with respect to vector layers (polygon/line)
- [x] Raster to/from polygons conversion 
- [x] Compute horizon obstruction from DEM


## Requirements
In theory, it should run with any equal or earlier version of the following libraries. Please report any issue you might cope with.

* `` cpc.geogrids == 0.2.3 ``
* `` elevation >= 1.0.5 ``
* `` fiona >= 1.7.13 ``
* `` gdal >= 2.2.4 ``
* `` geopandas >= 0.3.0 ``
* `` matplotlib >= 2.2.3 ``
* `` networkx >= 2.1 ``
* `` numpy >= 1.14.3 ``
* `` metis == 0.2a4``
* `` pandas >= 0.23.4``
* `` progressbar2 >= 3.38.0 ``
* `` pyproj >= 1.9.5 ``
* `` rasterio >= 0.36.0 ``
* `` rtree >= 0.8.3 ``
* `` scipy >= 1.1.0 ``
* `` shapely >= 1.6.4 ``


## Install
Pip installation should normally take care of everything for you.

### Using PyPi

The easiest may to install GisTools is using ``pip`` in a terminal
```
$ pip install gis_tools
```


## Examples

### Example 1

Use ``DigitalElevationModel``, ``PolygonLayer`` and ``ZonalStatistics`` classes to retrieve average slope within each polygon of a shapefile
```
>>> dem = gistools.raster.DigitalElevationModel("path/to/dem.tif")
>>> slope = dem.compute_slope()
>>> layer = gistools.layer.PolygonLayer("path/to/layer.shp")
>>> zonal_stat = gistools.stats.ZonalStatistics(slope, layer, is_surface_weighted=False, all_touched=True)
>>> average = zonal_stat.mean()
```

### Example 2

Extract polygons from contour values in raster
```
>>> from gistools.raster import RasterMap
>>> raster = RasterMap("path/to/raster.tif", no_data_value=-9999)
>>> layer = raster.contour(0.04, False).polygonize("attribute name").to_crs(epsg=4326)
```

### Example 3

Build DEM tile by downloading from CGIAR website and save to file
```
>>> from gistools import DigitalElevationModel
>>> dem = DigitalElevationModel.from_cgiar_online_database((8, 38, 14, 42))
>>> dem.to_file("path/to/dem.tif")
```
