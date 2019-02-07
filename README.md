# GisTools
Some geographical tools for python developers

## Introduction
GisTools is a small Python library for performing geographical computations.

## Requirements
In theory, it should run with any equal or earlier version of the following libraries. Please report any issue you might cope with.

* `` cpc.geogrids >= 0.2.3 ``
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
Be careful when you install GisTools dependencies. If you're not sure about it, prefer the conda installation which should take care of everything for you, except ``cpc.geogrids`` and ``elevation`` whose installation is described below.

### Using PyPi

You may install GisTools using ``pip`` in a terminal
```
$ pip install gis_tools
```

### Using conda

GisTools is also installable through Anaconda
```
$ conda install -c benjaminpillot gis_tools
```

**Note:** 

Installing ``cpc.geogrids`` dependency
```
$ git clone https://github.com/noaa-nws-cpc/cpc.geogrids
$ cd cpc.geogrids/
$ make install
```

Installing ``elevation`` dependency (optional, only useful for downloading DEM tile from SRTM online databases)
```
$ pip install elevation
```

## Examples

### Example 1

Use ``DigitalElevationModel``, ``PolygonLayer`` and ``ZonalStatistics`` classes to retrieve average slope within each polygon of a shapefile
```
>>> dem = DigitalElevationModel("path/to/dem.tif")
>>> slope = dem.compute_slope()
>>> layer = PolygonLayer("path/to/layer.shp")
>>> zonal_stat = ZonalStatistics(slope, layer, is_surface_weighted=False, all_touched=True)
>>> average = zonal_stat.mean()
```

### Example 2
```
>>> raster = RasterMap("path/to/raster.tif", no_data_value=-9999)
>>> polygons = PolygonLayer("path/to/polygon.shp").to_crs(epsg=4326)
>>> layer = raster.contour(0.04, False).polygonize("attribute name").to_crs(epsg=4326)
```

