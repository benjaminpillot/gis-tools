from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='gis_tools',
      version='0.15.7',
      description='Some geographical tools for Python developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/benjamin2b/gis_tools',
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
                        "numpy>=1.14.3",
                        "pandas>=0.24.2",
                        "pyproj>=1.9.5",
                        "rasterio>=1.0.18",
                        "rtree>=0.8.3",
                        "scipy>=1.1.0",
                        "shapely>=1.6.4"],
      license='GNU GPL v3.0',
      packages=find_packages(),
      zip_safe=False)
