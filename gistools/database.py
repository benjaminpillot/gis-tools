# -*- coding: utf-8 -*-

""" Manage database connections and instances (postgis, etc.)

This module intends to make data reading from spatial databases with gistools easy
"""
import warnings

from geopandas import GeoDataFrame
from gistools.exceptions import SpatialDatabaseWarning, SpatialDatabaseError, GeoLayerError
from gistools.layer import PolygonLayer, PointLayer, LineLayer
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

from gistools.utils.check.value import check_string


class SpatialDatabase:
    """ Spatial database class based on Postgresql

    """

    session = None
    layer_class = {'point': PointLayer, 'line': LineLayer, 'polygon': PolygonLayer}

    def __init__(self, user, passwd, ip, db_name):
        """ Initialize spatial database from db string connection

        :param user: database user
        :param passwd: database password
        :param ip: database server ip address
        :param db_name: database name
        """

        self.engine = create_engine('postgresql://%s:%s@%s/%s' % (user, passwd, ip, db_name), echo=True)
        self.db_name = db_name
        self.inspector = inspect(self.engine)

    def automap(self, schema=None):
        """ Automatically map database into sqlalchemy ORM model

        """

        metadata = MetaData()
        metadata.reflect(self.engine, only=self.get_table_names())
        base = automap_base(metadata=metadata)
        base.prepare()

        for table_name in self.get_table_names(schema):
            try:
                self.__setattr__(table_name.capitalize(), base.classes.__getattr__(table_name))
            except AttributeError:
                warnings.warn("Invalid or missing table '%s'" % table_name, SpatialDatabaseWarning)

        self.session = Session(self.engine)

    def get_schema_names(self):
        """ Get schema names in database

        :return:
        """
        return self.inspector.get_schema_names()

    def get_table_names(self, schema=None):
        """ Get table names in database, according to schema

        :param schema:
        :return:
        """
        return self.inspector.get_table_names(schema)

    def table_to_layer(self, table_name, schema=None, geom_type=None, bounds=None, polygon_extent=None):
        """ Convert table from database to GeoLayer instance

        :param table_name: name of table (case sensitive)
        :param schema: database schema (case sensitive)
        :param geom_type: geometry type
        :param bounds: bounding box (x_min, y_min, x_max, y_max)
        :param polygon_extent: shapely polygon
        :return:
        """
        if schema is None:
            schema = "public"

        if bounds is not None and polygon_extent is None:
            sql_string = f'SELECT * FROM "{schema}"."{table_name}" WHERE "{schema}"."{table_name}".geom && ' \
                         f'ST_MakeEnvelope({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})'
        elif polygon_extent is not None and bounds is None:
            sql_string = f'SELECT * FROM "{schema}"."{table_name}" WHERE ST_Within("{schema}"."{table_name}".geom, ' \
                         f'{polygon_extent})'
        else:
            sql_string = f'SELECT * FROM "{schema}"."{table_name}"'

        df = GeoDataFrame.from_postgis(sql_string, self.engine)

        if table_name in self.get_table_names(schema) and geom_type is None:
            try:
                layer = PolygonLayer(df, name=table_name)
            except GeoLayerError:
                try:
                    layer = LineLayer(df, name=table_name)
                except GeoLayerError:
                    layer = PointLayer(df, name=table_name)
        elif table_name in self.get_table_names(schema) and geom_type is not None:
            try:
                geom_type = check_string(geom_type, ("point", "line", "polygon"))
            except ValueError:
                raise SpatialDatabaseError("Invalid geometry type '%s'. Must be 'point', 'line' or 'polygon'." %
                                           geom_type)
            layer = self.layer_class[geom_type](df, name=table_name)
        else:
            raise SpatialDatabaseError("No table named '%s' in database '%s'" % (table_name, self.db_name))

        return layer
