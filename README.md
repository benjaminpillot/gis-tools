# geotools
Some geographical tools for python developers

## Introduction
GeoTools is a small Python library for performing geographical computations.

## Requirements
In theory, it should run with any equal or earlier version of the following libraries. However, it is possible to experience some trouble with some earlier versions of the rasterio package. Please report any issue you might cope with.

* `` cpc.geogrids >= 0.2.3 ``
* `` geopandas >= 0.3.0 ``
* `` networkx >= 2.1 ``
* `` numpy >= 1.14.3 ``
* `` osgeo >= 2.2.4 ``
* `` pyproj >= 1.9.5 ``
* `` rasterio >= 0.36.0 ``
* `` rtree >= 0.8.3 ``
* `` shapely >= 1.6.4 ``

## Install
You may install GisTools using ``pip`` in a terminal
```
$ pip install gis_tools
```
#### Install cpc.geogrids package
```
$ git clone https://github.com/noaa-nws-cpc/cpc.geogrids
$ cd cpc.geogrids/
$ make install
