# GisTools
Some geographical tools for python developers

## Introduction
GisTools is a small Python library for performing geographical computations. Typically, it gathers different tools from well-known libraries such as gdal, rasterio, geopandas, fiona and shapely.
It allows easily mixing operations between vectors and raster maps (multi-band raster are not supported at the moment).

## Basic operations
GisTools allows some of the following operations:
- [x] Fast polygon intersection and split
- [x] Polygon partition based on graph theory (requires [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) package)
- [x] Basic networking (shortest path)
- [x] Downloading DEM from online databases
- [x] Extract raster statistics with respect to vector layers (polygon/line)
- [x] Raster to/from polygons conversion 
- [x] Compute horizon obstruction from DEM


## Requirements
See ``requirements.txt``.


## Install
Pip installation should normally take care of everything for you.

### Using PIP

The easiest way to install GisTools is using ``pip`` in a terminal
```
$ pip install gis-tools
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

### Example 4

Split a polygon layer into sub-polygons of equal area with respect to 
honeycomb mesh (requires [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) package)
```
>>> polygon_layer = gistools.layer.PolygonLayer("path/to/layer.geojson")
>>> new_partitioned_layer = polygon_layer.partition(threshold=2000, disaggregation_factor=20, 
                                                    split_method="hexana", contig=True)
```
