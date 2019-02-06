from setuptools import setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='gis_tools',
      version='0.15.01',
      description='Some geographical tools for Python developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/benjamin2b/gis_tools',
      author='Benjamin Pillot',
      author_email='benjaminpillot@riseup.net',
      license='GNU GPL v3.0',
      packages=['gistools', 'utils', 'utils.check', 'utils.sys', 'toolset', 'examples', 'examples.topography',
                'examples.zonal_stat'],
      zip_safe=False)
