from setuptools import setup, find_packages

import gistools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='gis_tools',
      version=gistools.__version__,
      description='Some geographical tools for Python developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/benjaminpillot/gis-tools',
      author='Benjamin Pillot',
      author_email='benjaminpillot@riseup.net',
      install_requires=["cpc.geogrids==0.2.3",
                        "elevation>=1.0.5",
                        "fiona>=1.7.13",
                        "gdal>=2.2.4",
                        "geopandas>=0.4.0",
                        "greece-utils>=0.1.0"
                        "matplotlib>=3.0.3",
                        "metis>=0.2a4",
                        "networkx>=2.1",
                        "numba>=0.44.1",
                        "numpy>=1.14.3",
                        "pandas>=0.24.2",
                        "progressbar2>=3.39.3",
                        "pyproj==2.0.2",
                        "rasterio>=1.0.18",
                        "rdp>=0.8",
                        "rtree>=0.8.3",
                        "scipy>=1.1.0",
                        "shapely>=1.6.4"],
      python_requires='>=3',
      license='GNU GPL v3.0',
      packages=find_packages(),
      zip_safe=False)
