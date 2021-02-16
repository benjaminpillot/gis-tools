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
        graph = max((graph.subgraph(c) for c in nx.connected_components(graph)), key=len)

    graph.graph["node_weight_attr"] = node_weight_attr
    _, parts = metis.part_graph(graph, nparts, tpwgts=tpweights,
                                ubvec=None, recursive=recursive, **metis_options)
    partition = [[] for _ in range(nparts)]
    for u, i in zip(graph, parts):
        partition[i].append(u)

    # Only return non-empty parts
    return [part for part in partition if part]


if __name__ == "__main__":
    from matplotlib import pyplot
    from gistools.layer import PolygonLayer
    test = PolygonLayer("/home/benjamin/Documents/PRO/PROJET_GREECE_OPSPV/001_DONNEES/data_greece"
                        "/Geo layers/Parc amazonien/enp_pn_s_973.shp")
    # test = PolygonLayer("/home/benjamin/Desktop/APUREZA/geocoding/04_Codes/01_CodeSaoSeb/admin_level_10.shp")
    test = test.to_crs(epsg=32723)
    m = test.partition(100000000, contig=True, ncuts=50, show_progressbar=True, objtype="cut")

    m.plot()

    pyplot.show()

