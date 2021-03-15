from setuptools import setup, find_packages

import gistools

with open("README.md", 'r') as fh:
    long_description = fh.read()

with open("requirements.txt") as req:
    install_req = req.read().splitlines()

setup(name='gis_tools',
      version=gistools.__version__,
      description='Some geographical tools for Python developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/benjaminpillot/gis-tools',
      author='Benjamin Pillot',
      author_email='benjaminpillot@riseup.net',
      install_requires=install_req,
      python_requires='>=3',
      license='MIT',
      packages=find_packages(exclude=("gistools/examples", "gistools/tests")),
      zip_safe=False)
