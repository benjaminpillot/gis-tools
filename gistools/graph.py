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
    # TODO: adapt code to networkx version 2.4 ! (The problem is about the metis module...)
    # see https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/components/connected.html
    # nx.connected_component_subgraphs does not exist anymore in version 2.4 !
    # graph = max(list([graph.subgraph(c).copy() for c in nx.connected_components(graph)]), key=len)

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


if __name__ == "__main__":
    from gistools.layer import PolygonLayer
    test = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_10.shp")
    test = test.to_crs(epsg=32723)
    m = test.partition(50000, contig=True, ncuts=2)

