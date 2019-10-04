# -*- coding: utf-8 -*-

""" Manage database connections and instances (postgis, etc.)

This module intends to make data reading from spatial databases with gistools easy
"""
from sqlalchemy import create_engine

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2019, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


class SpatialDatabase:

    def __init__(self, user, passwd, ip, db_name):

        self.engine = create_engine('postgresql://%s:%s@%s/%s' % (user, passwd, ip, db_name), echo=True)