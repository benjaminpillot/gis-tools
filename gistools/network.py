# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'

import networkx as nx
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString

from gistools.coordinates import r_tree_idx
from gistools.exceptions import EdgeError, NetworkError, RoadError, RoadNodeError
from gistools.layer import return_new_instance, LineLayer, PointLayer
from toolset.list import split_list_by_index
from utils.check import check_type, check_string, protected_property, type_assert, check_type_in_collection

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


# Division by zero
np.seterr(divide='ignore')

# Constants
SPEED_RATIO = {'m/s': 1, 'km/h': 3.6}
TIME_FORMAT = {'s': 1, 'm': 1/60, 'h': 1/3600}


def find_all_disconnected_edges_and_fix(edges, tolerance, method):
    """ Find all disconnected edges of network and fix it

    :param edges: Edge class instance
    :param tolerance: tolerance for considering "disconnected" an edge
    :param method: fix method
    :return:
    """
    while "There is still disconnected edges":
        new_edges = edges.find_disconnected_islands_and_fix(tolerance=tolerance, method=method)
        if len(new_edges) < len(edges):
            edges = new_edges
        else:
            break

    return new_edges


class Edge(LineLayer):
    """ Edge class

    Class for implementing edges
    in a geo network
    """
    from_to = protected_property('from_to')

    DEFAULT_DIRECTION = "two-ways"

    def __init__(self, edges, *args, **kwargs):
        """ Edge class constructor

        :param edges: geo-like file (e.g. shapefile) or geopandas data frame
        """
        super().__init__(edges, *args, **kwargs)

        # Set Edge specific attributes
        if "direction" not in self.attributes():
            self._gpd_df["direction"] = [self.DEFAULT_DIRECTION] * len(self)  # Set default direction (not directed)

        # Simplified edges
        self._from_to = [((geom.coords.xy[0][0], geom.coords.xy[1][0]), (geom.coords.xy[0][-1], geom.coords.xy[1][
            -1])) for geom in self.geometry]

        # Override point layer class attribute (To Edge is associated Node)
        self._point_layer_class = Node

    def _check_attr_and_dic(self, attr_name, dic):

        if attr_name not in self.attributes():
            raise EdgeError("Unknown attribute name '%s'" % attr_name)

        if any([val not in list(set(self[attr_name])) for val in dic.keys()]):
            raise EdgeError("Invalid key in '%s'" % dic)

    def find_disconnected_islands_and_fix(self, tolerance=None, method="delete"):
        """ Find disconnected components in network

        Find disconnected components/islands graphs in multi-graph
        and apply method (fix/reconnect, keep, delete) with respect
        to a given tolerance
        :param tolerance:
        :param method:
        :return:
        """
        method = check_string(method, {'reconnect_and_delete', 'reconnect_and_keep', 'delete'})
        undirected_graph = nx.MultiGraph()
        undirected_graph.add_edges_from([(from_to[0], from_to[1]) for from_to in self.from_to])
        sub_graphs = list(nx.connected_component_subgraphs(undirected_graph))
        main_component = max(sub_graphs, key=len)
        sub_graphs.remove(main_component)

        idx_edge = []
        if method == "delete":
            for graph in sub_graphs:
                for edge in graph.edges:
                    try:
                        idx_edge.append(self.from_to.index((edge[0], edge[1])))
                    except ValueError:
                        idx_edge.append(self.from_to.index((edge[1], edge[0])))

        elif method == 'reconnect_and_delete':
            pass
        elif method == 'reconnect_and_keep':
            pass

        return self.drop(self.index[idx_edge])

        # TODO: implement island reconnection and tolerance

    @return_new_instance
    def get_path(self, path):

        edge_path = gpd.GeoDataFrame(columns=self.attributes(), crs=self.crs)

        for i in range(len(path) - 1):
            try:
                edge_path = edge_path.append(self._gpd_df.loc[self._from_to.index((path[i], path[i + 1]))],
                                             ignore_index=True)
            except ValueError:
                edge_path = edge_path.append(self._gpd_df.loc[self._from_to.index((path[i + 1], path[i]))],
                                             ignore_index=True)

        return edge_path

    def get_nodes(self):
        """ Get nodes from edges

        :return:
        """
        from_node = [coords[0] for coords in self._from_to]
        to_node = [coords[1] for coords in self._from_to]

        return self._point_layer_class(
            gpd.GeoDataFrame(geometry=[Point(coord) for coord in list(set(from_node + to_node))], crs=self.crs))

    @return_new_instance
    def get_simplified(self):
        """ Return simplified edge

        Return simplified Edge instance, that is only with starting
        and ending coordinates of each road segment
        :return:
        """
        outdf = self._gpd_df.copy()
        outdf.geometry = [LineString(from_to) for from_to in self._from_to]

        return outdf

    @type_assert(attribute_name=str, direction_dic=dict)
    def set_direction(self, attribute_name, direction_dic):
        """ Set edge direction

        :param attribute_name: layer attribute from which direction must be derived
        :param direction_dic: (valid direction values: "two-ways", "one-way", "reverse", None)
        :return:
        """
        self._check_attr_and_dic(attribute_name, direction_dic)

        for key in direction_dic.keys():
            if direction_dic[key] not in ['two-ways', 'one-way', 'reverse', None]:
                raise EdgeError("'%s' is not a valid direction value" % direction_dic[key])
            self._gpd_df.loc[self[attribute_name] == key, "direction"] = direction_dic[key]

    def split_at_underlying_points(self, location, *args):
        """ Override parent class method

        Split corresponding attributes in addition
        to layer, i.e.
        :param location:
        :return:
        """
        output = super().split_at_underlying_points(location)
        if len(args) == 0:
            return output

        outputs = [output]
        for attr in args:
            split_attr = []
            for n, a in enumerate(attr):
                break_idx = [loc[1] for loc in location if loc[0] == n]
                if len(break_idx) == 0:
                    split_attr.append(a)
                else:
                    split_attr.extend(split_list_by_index(a, break_idx, include=False))
            outputs.append(split_attr)

        return tuple(outputs)

    @property
    def direction(self):
        return self["direction"]


class Node(PointLayer):
    """ Node class

    Class for implementing a set of nodes
    in a geo network
    """

    def __init__(self, nodes, *args, **kwargs):
        """ Node class constructor

        :param nodes:
        """
        super().__init__(nodes, *args, **kwargs)


class Road(Edge):
    """ Road class

    Class for implementing a set of roads in a geo network
    """

    DEFAULT_MAX_SPEED = 25
    DEFAULT_ROLLING_COEFFICIENT = 0.01
    DEFAULT_ROLLOVER_CRITERION = 0.15

    def __init__(self, roads, *args, **kwargs):
        """ Road class constructor

        :param roads:
        :param args:
        :param kwargs:
        """
        super().__init__(roads, *args, **kwargs)

        # Max speed on road segment
        if "max_speed" not in self.attributes():
            self._gpd_df["max_speed"] = [self.DEFAULT_MAX_SPEED] * len(self)

        # Rolling coefficient on road segment
        if "rolling_coefficient" not in self.attributes():
            self._gpd_df["rolling_coefficient"] = [self.DEFAULT_ROLLING_COEFFICIENT] * len(self)

        # Rollover criterion on road segment
        if "rollover_criterion" not in self.attributes():
            self._gpd_df["rollover_criterion"] = [self.DEFAULT_ROLLOVER_CRITERION] * len(self)

        # Override point layer class
        self._point_layer_class = RoadNode

    @type_assert(attr_name=str, speed_dic=dict)
    def set_max_speed(self, attr_name, speed_dic, speed_format='km/h'):
        """ Set road max allowed speed

        :param attr_name:
        :param speed_dic:
        :param speed_format:
        :return:
        """
        check_string(speed_format, ('m/s', 'km/h'))
        self._check_attr_and_dic(attr_name, speed_dic)

        for key in speed_dic.keys():
            if not isinstance(speed_dic[key], (float, int)):
                raise RoadError("Speed value must be numeric but is '%s'" % type(speed_dic[key]))
            self._gpd_df.loc[self[attr_name] == key, "max_speed"] = speed_dic[key] / SPEED_RATIO[speed_format]

    @type_assert(attr_name=str, rolling_coeff_dic=dict)
    def set_rolling_coefficient(self, attr_name, rolling_coeff_dic):
        """ Set road rolling coefficient

        :param attr_name:
        :param rolling_coeff_dic:
        :return:
        """
        self._check_attr_and_dic(attr_name, rolling_coeff_dic)

        for key in rolling_coeff_dic.keys():
            if not isinstance(rolling_coeff_dic[key], float):
                raise RoadError("Rolling coefficient must be a float but is '%s'" % type(rolling_coeff_dic[key]))
            self._gpd_df.loc[self[attr_name] == key, "rolling_coefficient"] = rolling_coeff_dic[key]

    @type_assert(attr_name=str, rollover_criterion_dic=dict)
    def set_rollover_criterion(self, attr_name, rollover_criterion_dic):
        """ Set road rollover criterion

        :param attr_name:
        :param rollover_criterion_dic:
        :return:
        """
        self._check_attr_and_dic(attr_name, rollover_criterion_dic)

        for key in rollover_criterion_dic.keys():
            if not isinstance(rollover_criterion_dic[key], float):
                raise RoadError("Rollover criterion must be a float but is '%s'" % type(rollover_criterion_dic[key]))
            self._gpd_df.loc[self[attr_name] == key, "rollover_criterion"] = rollover_criterion_dic[key]

    @property
    def max_speed(self):
        return self["max_speed"]

    @property
    def rolling_coefficient(self):
        return self["rolling_coefficient"]

    @property
    def rollover_criterion(self):
        return self["rollover_criterion"]


class RoadNode(Node):
    """ RoadIntersection class

    Class for implementing road intersections (inherits from Node)
    """

    _check_attr_and_dic = Edge._check_attr_and_dic

    DEFAULT_MAX_SPEED = 0

    def __init__(self, nodes, *args, **kwargs):
        super().__init__(nodes, *args, **kwargs)

        if "max_speed" not in self.attributes():
            self._gpd_df["max_speed"] = [self.DEFAULT_MAX_SPEED] * len(self)

    def set_max_speed(self, attr_name, speed_dic, speed_format='km/h'):
        """ Set max allowed speed at intersection

        :param attr_name:
        :param speed_dic:
        :param speed_format:
        :return:
        """
        check_string(speed_format, ('m/s', 'km/h'))
        self._check_attr_and_dic(attr_name, speed_dic)

        for key in speed_dic.keys():
            if not isinstance(speed_dic[key], (float, int)):
                raise RoadNodeError("Speed value must be numeric but is '%s'" % type(speed_dic[key]))
            self._gpd_df.loc[self[attr_name] == key, "max_speed"] = speed_dic[key] / SPEED_RATIO[speed_format]

    @property
    def max_speed(self):
        return self["max_speed"]


class Network:
    """ Network base class

    Use this class to implement sub-class network
    geometry (e.g. from shapefile) and apply
    corresponding tools
    """
    edges = protected_property("edges")
    nodes = protected_property("nodes")

    def __init__(self, edges, nodes, match_edge_nodes=True, tolerance=1):
        """ Network class constructor

        :param edges: Edge instance
        :param nodes: Node instance
        :param match_edge_nodes: Boolean --> match edge nodes with respect to tolerance
        :param tolerance: distance tolerance for considering nodes and edge nodes the same (in m)
        """
        check_type(edges, Edge, nodes, Node)
        self._graph = nx.MultiDiGraph()
        self._edges = edges
        self._nodes = nodes

        # Retrieve edge nodes corresponding to nodes
        if match_edge_nodes:
            edge_nodes = self._edges.get_nodes()
            distance, nn = nodes.distance_and_nearest_neighbor(edge_nodes)
            self._nodes["geometry"] = [edge_nodes.geometry[n] for n in nn]
            self._nodes = self._nodes[distance <= tolerance]

    def build_graph(self, weight_one_way=None, weight_return=None):
        """ Build corresponding graph

        :param weight_one_way: array of weight values for edge in one_way direction
        :param weight_return: array of weight values for edge in reverse direction
        :return:
        """
        if weight_one_way is None:
            weight_one_way = self._edges.length
        if weight_return is None:
            weight_return = self._edges.length

        if len(weight_one_way) != len(self._edges) or len(weight_return) != len(self._edges):
            raise NetworkError("Input argument(s) must have the same length as network edges")

        weight = []
        from_node = []
        to_node = []
        for idx, coords in enumerate(self._edges.from_to):
            if self._edges.direction[idx] != "reverse":
                weight.append(weight_one_way[idx])
                from_node.append(coords[0])
                to_node.append(coords[1])
            if self._edges.direction[idx] != "one-way":
                weight.append(weight_return[idx])
                from_node.append(coords[1])
                to_node.append(coords[0])

        # Set graph
        self._graph.clear()
        self._graph.add_weighted_edges_from([(from_n, to_n, w) for from_n, to_n, w in zip(from_node, to_node, weight)])

        return self

    # TODO: must redefine the following method
    # def find_disconnected_islands_and_fix(self, tolerance=None, method="delete"):
    #     """ Find disconnected components in network
    #
    #     Find disconnected components/islands graphs in multi-graph
    #     and apply method (fix/reconnect, keep, delete) with respect
    #     to a given tolerance
    #     :param tolerance: tolerance for considering an island "disconnected"
    #     :param method:
    #     :return:
    #     """
    #     edges = self._edges.copy()
    #     while "There is still disconnected islands":
    #         new_edges = edges.find_disconnected_islands_and_fix(tolerance, method)
    #         if len(new_edges) < len(edges):
    #             edges = new_edges
    #         else:
    #             break
    #
    #     return type(self)(new_edges, self._nodes)

    def get_minimum_distance_to_network(self, layer):
        """ get minimum distance from given layer to network

        :param layer:
        :return:
        """
        distance_to_edge = layer.distance(self.edges)
        distance_to_node = layer.distance(self.nodes)

        return np.minimum(distance_to_edge, distance_to_node)

    @type_assert(node_start=Point, node_end=Point)
    def get_shortest_path(self, node_start, node_end):
        """ Get shortest path between 2 nodes using Dijkstra algorithm

        :param node_start: shapely Point
        :param node_end: shapely Point
        :return: Edge instance of the path
        """
        if node_start not in self._nodes.geometry or node_end not in self._nodes.geometry:
            raise EdgeError("Either source or destination node is invalid")

        node_start = (node_start.x, node_start.y)
        node_end = (node_end.x, node_end.y)

        if node_start == node_end:
            return []  # Empty path

        try:
            path = nx.dijkstra_path(self._graph, node_start, node_end)
        except nx.NetworkXNoPath:
            print("No available path between node {} and node {}".format(node_start, node_end))
            return None
        else:
            return self._edges.get_path(path)

    @type_assert(node_start=Point, node_end=Point)
    def get_shortest_path_length(self, node_start, node_end, method: str = "networkx"):
        """ Get dijkstra shortest path length

        :param node_start: shapely Point
        :param node_end: shapely Point
        :param method: {'internal', 'networkx'}
        :return: length of path in m
        """
        check_string(method, ("internal", "networkx"))
        if method == "internal":
            edge_path = self.get_shortest_path(node_start, node_end)
            length = 0
            if edge_path is not None and edge_path != []:
                for edge in edge_path.geometry:
                    length += edge.length
            elif edge_path is None:
                return None
        else:
            node_start = (node_start.x, node_start.y)
            node_end = (node_end.x, node_end.y)
            length = nx.dijkstra_path_length(self._graph, node_start, node_end)

        return length

    def get_all_shortest_paths(self):
        """ Get shortest paths between all graph nodes

        :return:
        """
        return nx.all_pairs_dijkstra_path(self._graph)

    def get_all_shortest_path_lengths(self):
        """ Get shortest path lengths between all graph nodes

        :return:
        """
        return nx.all_pairs_dijkstra_path_length(self._graph)

    @type_assert(source_node=Point)
    def get_all_shortest_paths_from_source(self, source_node):
        """ Get all paths from one source node using Dijkstra

        :param source_node:
        :return:
        """
        if source_node not in self._nodes.geometry:
            raise NetworkError("Source node is invalid")

        source_node = (source_node.x, source_node.y)

        return nx.single_source_dijkstra_path(self._graph, source_node)

    @type_assert(source_node=Point)
    def get_all_shortest_path_lengths_from_source(self, source_node):
        """

        :param source_node:
        :return:
        """
        if source_node not in self._nodes.geometry:
            raise NetworkError("Source node is invalid")

        source_node = (source_node.x, source_node.y)

        return nx.single_source_dijkstra_path_length(self._graph, source_node)

    @type_assert(source_node=Point)
    def get_shortest_paths_from_source(self, source_node, target_nodes):
        """ Get multiple shortest paths from single source using Dijkstra

        :param source_node:
        :param target_nodes:
        :return:
        """
        try:
            check_type_in_collection(target_nodes, Point)
        except TypeError:
            raise NetworkError("'%s' must be a collection of Point instances" % target_nodes)

        paths = []
        all_paths = self.get_all_shortest_paths_from_source(source_node)
        for target in target_nodes:
            target = (target.x, target.y)
            if target in all_paths.keys():
                paths.append(self._edges.get_path(all_paths[target]))

        return paths

    @type_assert(source_node=Point)
    def get_shortest_path_lengths_from_source(self, source_node, target_nodes):
        """

        :param source_node:
        :param target_nodes:
        :return:
        """
        try:
            check_type_in_collection(target_nodes, Point)
        except TypeError:
            raise NetworkError("'%s' must be a collection of Point instances" % target_nodes)

        path_lengths = []
        all_path_lengths = self.get_all_shortest_path_lengths_from_source(source_node)
        for target in target_nodes:
            target = (target.x, target.y)
            if target in all_path_lengths.keys():
                path_lengths.append(all_path_lengths[target])

        return path_lengths

    def get_shortest_path_matrix(self):
        """ Get shortest path matrix

        Compute shortest path length between all
        starting and ending nodes
        :return:
        """
        shortest_path = np.full((len(self._nodes), len(self._nodes)), np.nan)
        edge_nodes = self._get_nearest_edge_node()
        for i, geom_from in enumerate(edge_nodes.geometry):
            for n, geom_to in enumerate(edge_nodes.geometry):
                shortest_path[i, n] = self._edges.get_dijkstra_path_length(geom_from, geom_to)

        return shortest_path

    def plot(self, edge_color="blue", node_color="red"):
        """

        :param edge_color:
        :param node_color:
        :return:
        """
        self._edges.plot(layer_color=edge_color)
        self._nodes.plot(layer_color=node_color)

    def _get_nearest_edge_node(self):
        """

        :return:
        """
        nodes = self._edges.get_nodes()
        idx = r_tree_idx(nodes.geometry)
        edge_nodes = []
        for geom in self._nodes.geometry:
            nn = list(idx.nearest(geom.bounds, 1))
            edge_nodes.append(nodes.geometry[nn[0]])

        edge_nodes = Node(gpd.GeoDataFrame(geometry=edge_nodes, crs=self._edges.crs))
        return edge_nodes


class RoadNetwork(Network):
    """ Road network class

    """

    roads = protected_property("edges")

    def __init__(self, roads, nodes, *args, **kwargs):
        """

        :param roads: Road instance
        :param nodes: road nodes
        """
        check_type(roads, Road, nodes, RoadNode)

        super().__init__(roads, nodes, *args, **kwargs)

    def fuel_consumption(self, gross_hp, vehicle_weight, vehicle_frontal_area=7.92, engine_efficiency=0.4,
                         fuel_energy_density=35, uphill_hp=0.8, downhill_hp=0.6, drag_resistance=0.35,
                         mass_correction_factor=1.05, acceleration_rate=1.5 * 0.3048, deceleration_rate=-9.5 * 0.3048):
        """ Compute fuel consumption on road segments

        :param vehicle_weight:
        :param gross_hp:
        :param vehicle_frontal_area:
        :param engine_efficiency:
        :param fuel_energy_density: fuel efficiency as L/MJ
        :param uphill_hp:
        :param downhill_hp:
        :param drag_resistance:
        :param mass_correction_factor:
        :param acceleration_rate:
        :param deceleration_rate:
        :return:
        """

        slope = [self.roads.slope_of_geometry(i, slope_format="degree") for i in range(len(self.roads))]
        r_curvature = [self.roads.radius_of_curvature(i) for i in range(len(self.roads))]
        road_length = [self.roads.length_xyz_of_geometry(i) for i in range(len(self.roads))]

        # Maximum limited speed
        v_max_one_way, v_max_reverse = \
            self._get_max_limited_speed(slope, r_curvature, vehicle_weight, gross_hp, uphill_hp, downhill_hp)

        # Maximum speed at intersection
        v_in_max, v_out_max = self._get_velocity_at_intersection()

        # Parameters
        rho_air = 1.225

        # Compute fuel consumption
        fuel_demand = {'one-way': [], 'reverse': []}

        for n, row in self.roads.iterrows():

            # Travel time and distance of acceleration
            t_time_one_way, d_a_one_way = self._get_travel_time_and_distance_of_acceleration(
                v_max_one_way[n], road_length[n], v_in_max[n], v_out_max[n], acceleration_rate, deceleration_rate)
            t_time_reverse, d_a_reverse = self._get_travel_time_and_distance_of_acceleration(
                v_max_reverse[n], road_length[n], v_out_max[n], v_in_max[n], acceleration_rate, deceleration_rate)

            # Travel time (for mean velocity over road segment)
            v_mean_one_way = road_length[n] / t_time_one_way
            v_mean_reverse = road_length[n] / t_time_reverse

            # Energy demand
            u_r = self.roads.rolling_coefficient[n] * vehicle_weight * 9.81 * np.cos(slope[n] * np.pi / 180) * \
                road_length[n]
            u_a_one_way = 0.5 * rho_air * vehicle_frontal_area * drag_resistance * v_mean_one_way ** 2 * road_length[n]
            u_a_reverse = 0.5 * rho_air * vehicle_frontal_area * drag_resistance * v_mean_reverse ** 2 * road_length[n]
            u_i_one_way = mass_correction_factor * vehicle_weight * acceleration_rate * d_a_one_way
            u_i_reverse = mass_correction_factor * vehicle_weight * acceleration_rate * d_a_reverse
            u_g_one_way = vehicle_weight * 9.81 * np.sin(slope[n] * np.pi / 180) * road_length[n]
            u_g_reverse = vehicle_weight * 9.81 * np.sin(-slope[n] * np.pi / 180) * road_length[n]

            fuel_demand["one-way"].append(np.maximum(0, (u_r + u_a_one_way + u_i_one_way + u_g_one_way) * 1e-6 / (
                    fuel_energy_density * engine_efficiency)))
            fuel_demand["reverse"].append(np.maximum(0, (u_r + u_a_reverse + u_i_reverse + u_g_reverse) * 1e-6 / (
                    fuel_energy_density * engine_efficiency)))

        return fuel_demand

    def travel_time(self, gross_hp, vehicle_weight, acceleration_rate=1.5 * 0.3048, deceleration_rate=-9.5 * 0.3048,
                    uphill_hp=0.8, downhill_hp=0.6, time_format='h'):
        """ Compute travel time for each road segment

        Compute travel time for each road element according
        to given parameters
        :param gross_hp: gross horse power of the vehicle
        :param vehicle_weight: weight of the vehicle
        :param acceleration_rate: positive acceleration value
        :param deceleration_rate: negative acceleration value (deceleration)
        :param uphill_hp: available horsepower on uphill road (%)
        :param downhill_hp: available horsepower on downhill road (%)
        :param time_format: format of output time (seconds, minutes, hours)
        :return:
        """
        travel_time = {'one-way': [], 'reverse': []}
        slope = [self.roads.slope_of_geometry(i, slope_format="degree") for i in range(len(self.roads))]
        r_curvature = [self.roads.radius_of_curvature(i) for i in range(len(self.roads))]
        road_length = [self.roads.length_xyz_of_geometry(i) for i in range(len(self.roads))]

        # Maximum limited speed
        v_max_one_way, v_max_reverse = \
            self._get_max_limited_speed(slope, r_curvature, vehicle_weight, gross_hp, uphill_hp, downhill_hp)

        # Maximum speed at intersection
        v_in_max, v_out_max = self._get_velocity_at_intersection()

        for v, d, v_in, v_out in zip(v_max_one_way, road_length, v_in_max, v_out_max):
            time, _ = self._get_travel_time_and_distance_of_acceleration(
                v, d, v_in, v_out, acceleration_rate, deceleration_rate)
            travel_time["one-way"].append(TIME_FORMAT[time_format] * time)

        for v, d, v_in, v_out in zip(v_max_reverse, road_length, v_in_max, v_out_max):
            time, _ = self._get_travel_time_and_distance_of_acceleration(
                v[::-1], d[::-1], v_out, v_in, acceleration_rate, deceleration_rate)
            travel_time["reverse"].append(TIME_FORMAT[time_format] * time)

        return travel_time

    def _get_velocity_at_intersection(self):
        """ Velocity in crossing intersections

        Define maximum allowed entering and exiting velocities for each road segment
        :return:
        """
        node_coords = [(x, y) for x, y in zip(self.nodes.geometry.x, self.nodes.geometry.y)]
        v_in = np.full(len(self.roads), 0)
        v_out = np.full(len(self.roads), 0)

        for from_to in self.roads.from_to:
            v_in[node_coords.index(from_to[0])] = self.nodes.max_speed[node_coords.index(from_to[0])]
            v_out[node_coords.index(from_to[1])] = self.nodes.max_speed[node_coords.index(from_to[1])]

        return v_in, v_out

    #################
    # Private methods

    def _get_max_limited_speed(self, slope, r_curvature, vehicle_weight, gross_hp, uphill_hp, downhill_hp):
        """ Get maximum limited speed on road segments

        :param slope:
        :param r_curvature:
        :param vehicle_weight:
        :param gross_hp:
        :param uphill_hp:
        :param downhill_hp:
        :return:
        """

        # Maximum speed due to slope (1 mechanical hp = 745.699872 W)
        ehp_uphill = gross_hp * uphill_hp * 745.699872
        ehp_downhill = gross_hp * downhill_hp * 745.699872
        v_slope_one_way = []
        v_slope_reverse = []
        v_radius = []
        for n, row in self.roads.iterrows():
            v_one_way = np.zeros(len(slope[n]))
            v_reverse = np.zeros(len(slope[n]))
            grade_resistance = 9.81 * vehicle_weight * np.sin(np.fabs(slope[n]) * np.pi / 180)
            rolling_resistance = 9.81 * self.roads.rolling_coefficient[n] * vehicle_weight * np.cos(slope[n] * np.pi /
                                                                                                    180)
            v_one_way[slope[n] < 0] = ehp_downhill / np.maximum((grade_resistance[slope[n] < 0] - rolling_resistance[
                slope[n] < 0]), 0)
            v_one_way[slope[n] >= 0] = ehp_uphill / (grade_resistance[slope[n] >= 0] + rolling_resistance[slope[n] >=
                                                                                                          0])
            v_reverse[slope[n] > 0] = ehp_downhill / np.maximum((grade_resistance[slope[n] > 0] - rolling_resistance[
                slope[n] > 0]), 0)
            v_reverse[slope[n] <= 0] = ehp_uphill / (grade_resistance[slope[n] <= 0] + rolling_resistance[slope[n] <=
                                                                                                          0])
            v_slope_one_way.append(v_one_way)
            v_slope_reverse.append(v_reverse)
            v_radius.append((self.roads.rollover_criterion[n] * r_curvature[n] * 9.81) ** 0.5)

        # Get maximum limiting speed, i.e. minimum among all previous values
        v_max_one_way = [np.minimum(np.minimum(v_r, v_s), v_limit)
                         for v_r, v_s, v_limit in zip(v_radius, v_slope_one_way, self.roads.max_speed)]
        v_max_reverse = [np.minimum(np.minimum(v_r, v_s), v_limit)
                         for v_r, v_s, v_limit in zip(v_radius, v_slope_reverse, self.roads.max_speed)]

        return v_max_one_way, v_max_reverse

    @staticmethod
    def _get_travel_time_and_distance_of_acceleration(v_max, road_segment_length, v_in_max, v_out_max, a_1, a_2):
        """ Get travel time on road segment

        :param v_max: maximum limited speed on road segment
        :param road_segment_length: length of road segment
        :param v_in_max:
        :param v_out_max:
        :param a_1: acceleration rate
        :param a_2: deceleration rate
        :return:
        """

        v = np.concatenate([[v_in_max], v_max[1:], [v_out_max]])
        t_time = np.zeros(len(v_max))
        d_a = np.zeros(len(v_max))  # Distance of acceleration

        tol = 0.01  # Tolerance for comparing d <= s --> d must be <= s + tolerance (In order to avoid too much
        # backward in the while loop, as well as floating errors where d is not exactly equal to s at 1e-10)
        n = 0
        while n <= len(v_max) - 1:
            v_in = v[n]
            v_fn = v[n + 1]
            s = road_segment_length[n]
            v_m = v_max[n]

            d_1 = (v_m ** 2 - v_in ** 2) / (2 * a_1)
            d_2 = (v_fn ** 2 - v_m ** 2) / (2 * a_2)
            if v_m > v_in and v_m > v_fn:
                if v_fn >= v_in:
                    d = (v_fn ** 2 - v_in ** 2) / (2 * a_1)
                else:
                    d = (v_fn ** 2 - v_in ** 2) / (2 * a_2)
                if d_1 + d_2 <= s:
                    t_time[n] = (v_m - v_in) / a_1 + (v_fn - v_m) / a_2 + (s - (d_1 + d_2)) / v_m
                    d_a[n] = d_1
                    v[n + 1] = v_fn
                    n += 1
                else:
                    if d <= s + tol:
                        v_min = ((2 * s * a_1 * a_2 + a_2 * v_in ** 2 - a_1 * v_fn ** 2) / (a_2 - a_1)) ** 0.5
                        t_time[n] = (v_min - v_in) / a_1 + (v_fn - v_min) / a_2
                        d_a[n] = (v_min ** 2 - v_in ** 2) / (2 * a_1)
                        v[n + 1] = v_fn
                        n += 1
                    else:
                        if v_fn >= v_in:
                            v_front = (v_in ** 2 + 2 * a_1 * s) ** 0.5
                            t_time[n] = (v_front - v_in) / a_1
                            d_a[n] = s
                            v[n + 1] = v_front
                            n += 1
                        else:
                            v[n] = (v_fn ** 2 - 2 * a_2 * s) ** 0.5
                            n -= 1 if n > 0 else 0
            elif v_fn < v_m <= v_in:
                if d_2 <= s + tol:
                    t_time[n] = (v_fn - v_m) / a_2 + (s - d_2) / v_m
                    v[n + 1] = v_fn
                    n += 1
                else:
                    v[n] = (v_fn ** 2 - 2 * a_2 * s) ** 0.5
                    n -= 1 if n > 0 else 0
            elif v_in < v_m <= v_fn:
                if d_1 <= s:
                    t_time[n] = (v_m - v_in) / a_1 + (s - d_1) / v_m
                    d_a[n] = d_1
                    v[n + 1] = v_m
                else:
                    v_front = (v_in ** 2 + 2 * a_1 * s) ** 0.5
                    t_time[n] = (v_front - v_in) / a_1
                    d_a[n] = s
                    v[n + 1] = v_front
                n += 1
            elif v_m <= v_in and v_m <= v_fn:
                t_time[n] = s / v_m
                v[n + 1] = v_m
                n += 1

        return t_time, d_a


class ElectricalGrid(Network):
    pass


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils.sys.timer import Timer
    road = Road("/home/benjamin/Documents/Data/Geo layers/Road network/roads.shp")
    road_nodes = RoadNode("/home/benjamin/Documents/Data/Geo layers/Road network/nodes.shp")
    road_nodes.set_max_speed("NATURE", {'Carrefour simple': 30, 'Carrefour aménagé': 40})
    road.set_max_speed("ETAT", {'Non revêtu': 70, 'Revêtu': 90, 'Sentier': 50, "Chemin d'exploitation": 40})
    road.set_rolling_coefficient("ETAT", {'Non revêtu': 0.015, 'Revêtu': 0.01, 'Sentier': 0.02,
                                          "Chemin d'exploitation": 0.02})
    road.set_rollover_criterion("ETAT", {'Non revêtu': 0.18, 'Revêtu': 0.2, 'Sentier': 0.15,
                                         "Chemin d'exploitation": 0.15})
    road.set_direction('SENS', {'Double sens': 'two-ways', 'Sens inverse': 'reverse', 'Sens unique': 'one-way'})
    #
    road_network = RoadNetwork(road, road_nodes)
    #
    with Timer() as t:
        # fuel_demand_test = road_network.fuel_consumption(gross_hp=50, vehicle_weight=900, vehicle_frontal_area=2.52)
        fuel_demand_test = road_network.travel_time(gross_hp=100, vehicle_weight=900)
    # t_time = road_network.travel_time('ETAT', {'Non revêtu': 70, 'Revêtu': 90, 'Sentier': 50,
    #                                            "Chemin d'exploitation": 40}, 'ETAT',
    #                                   {'Non revêtu': 0.015, 'Revêtu': 0.01, 'Sentier': 0.02,
    #                                    "Chemin d'exploitation": 0.02}, acceleration_rate=1.5, gross_hp=500,
    #                                   vehicle_weight=30000, v_in_max=v_in, v_out_max=v_out, speed_format='km/h')
    print(t)

    # tot_time = {'one-way': [np.sum(t) for t in t_time["one-way"]], 'reverse': [np.sum(t) for t in t_time["reverse"]]}
    # road_network.build_graph(tot_time['one-way'], tot_time['reverse'])
    tot_fuel_demand = {'one-way': [np.sum(f) for f in fuel_demand_test['one-way']], 'reverse':
                       [np.sum(f) for f in fuel_demand_test['reverse']]}
    road_network.build_graph(tot_fuel_demand['one-way'], tot_fuel_demand['reverse'])

    _target_nodes = [road_nodes.geometry[234], road_nodes.geometry[456], road_nodes.geometry[765]]
    path_test_1 = road_network.get_shortest_path_lengths_from_source(road_nodes.geometry[6], _target_nodes)
    path_test_2 = road_network.get_shortest_paths_from_source(road_nodes.geometry[6], _target_nodes)

    path_test = road_network.get_shortest_path(road_nodes.geometry[6], road_nodes.geometry[765])
    path_fuel = road_network.get_shortest_path_length(road_nodes.geometry[6], road_nodes.geometry[765])
    road.plot(layer_color="blue")
    path_test.plot(layer_color="red")
    plt.show()
    print(path_fuel)
    print(np.sum(path_test.length))
    print(path_test_1)
    print([np.sum(p.length) for p in path_test_2])
    # print(len(path))
    # print("%.3f h" % road.get_dijkstra_path_length(nodes.geometry[6], nodes.geometry[765]))
    # road.build_graph()
    # print("%.3f m" % road.get_dijkstra_path_length(nodes.geometry[6], nodes.geometry[765]))
    # path = road.get_all_dijkstra_path_lengths_from_source(nodes.geometry[65])
    # print(t_)
    # print(type(road._graph))
    # path = road.get_dijkstra_path(nodes.geometry[345], nodes.geometry[489])
    # road.plot(layer_color="blue")
    # new_road.plot(layer_color="red")
    # path.plot(layer_color="red")
    # pyplot.show()
    # road = road.to_crs({'init': 'epsg:32622'})
