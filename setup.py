from setuptools import setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='gis_tools',
      version='0.14.4',
      description='Some geographical tools for Python developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/Benjamin2b/gis_tools',
      author='Benjamin Pillot',
      author_email='benjaminpillot@riseup.net',
      license='GNU GPL v3.0',
      packages=['gis_tools', 'utils', 'utils.check', 'utils.sys', 'toolset'],
      zip_safe=False)
