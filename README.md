# GisTools
Some geographical tools for python developers

## Introduction
GisTools is a small Python library for performing geographical computations.

## Requirements
In theory, it should run with any equal or earlier version of the following libraries. Please report any issue you might cope with.

* `` cpc.geogrids >= 0.2.3 ``
* `` fiona >= 1.7.13 ``
* `` geopandas >= 0.3.0 ``
* `` matplotlib >= 2.2.3 ``
* `` networkx >= 2.1 ``
* `` numpy >= 1.14.3 ``
* `` gdal >= 2.2.4 ``
* `` pyproj >= 1.9.5 ``
* `` rasterio >= 0.36.0 ``
* `` rtree >= 0.8.3 ``
* `` shapely >= 1.6.4 ``

## Install
Be careful when you install GisTools dependencies.

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

Installing cpc.geogrids dependency
```
$ git clone https://github.com/noaa-nws-cpc/cpc.geogrids
$ cd cpc.geogrids/
$ make install
```

Installing network-metis dependency
```
$ git clone https://github.com/networkx/networkx-metis.git
$ cd networkx-metis
$ python setup.py install
```

## Examples

Use ``DigitalElevationModel``, ``PolygonLayer`` and ``ZonalStatistics`` classes to retrieve average slope within each polygon of a shapefile
```
>>> dem = DigitalElevationModel("path/to/dem.tif")
>>> slope = dem.compute_slope()
>>> layer = PolygonLayer("path/to/layer.shp")
>>> zonal_stat = ZonalStatistics(slope, layer, is_surface_weighted=False, all_touched=True)
>>> average = zonal_stat.mean()
```

