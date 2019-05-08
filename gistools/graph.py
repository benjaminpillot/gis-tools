# -*- coding: utf-8 -*-

""" Graph module for graph computations

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
import networkx as nx
import warnings

from gistools.exceptions import ImportMetisWarning

try:
    import metis
except RuntimeError:  # No Metis DLL
    warnings.warn("Metis error. No graph partitioning will be available", ImportMetisWarning)

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
    # If contiguous partition is requested, only keep main contiguous graph component
    if metis_options["contig"]:
        graph = max(list(nx.connected_component_subgraphs(graph)), key=len)

    graph.graph["node_weight_attr"] = node_weight_attr
    _, parts = metis.part_graph(graph, nparts, tpwgts=tpweights, ubvec=None, recursive=recursive, **metis_options)
    partition = [[] for _ in range(nparts)]
    for u, i in zip(graph, parts):
        partition[i].append(u)

    # Only return non-empty parts
    return [part for part in partition if part]
