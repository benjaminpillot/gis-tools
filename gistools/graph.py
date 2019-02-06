# -*- coding: utf-8 -*-

""" Graph module for graph computations

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
import metis

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def part_graph(graph, nparts, node_weight_attr, tpweights, recursive, **metis_options):
    """ Partition graph

    :param graph:
    :param nparts:
    :param node_weight_attr:
    :param tpweights:
    :param recursive:
    :param metis_options:
    :return:
    """

    graph.graph["node_weight_attr"] = node_weight_attr
    _, parts = metis.part_graph(graph, nparts, tpwgts=tpweights, ubvec=None, recursive=recursive, **metis_options)
    partition = [[] for _ in range(nparts)]
    for u, i in zip(graph, parts):
        partition[i].append(u)

    return partition
