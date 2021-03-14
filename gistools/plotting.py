# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# import

# __all__ = []
# __version__ = '0.1'
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from gistools.utils.check.type import check_type


def plot_geolayer(gl, ax=None, attribute=None, layer_label: str = None, layer_color=None, labels: str = None,
                  marker=None, **kwargs):
    """ Plot geo layer

    :param gl:
    :param ax:
    :param attribute: attribute of the geo layer
    :param layer_label:
    :param layer_color:
    :param labels:
    :param marker:
    :param kwargs:
    :return:
    """

    def patch_point():
        return mlines.Line2D([0], [0], linestyle="none", marker=marker, color=layer_color, label=layer_label)

    def patch_line():
        return mlines.Line2D([], [], color=layer_color, label=layer_label)

    def patch_polygon():
        return mpatches.Patch(color=layer_color, label=layer_label)

    # Marker type for point geometry
    if marker is None:
        marker = "o"
    handles = []

    if ax is not None:
        try:
            handles = ax.handles
        except AttributeError:
            pass
    else:
        ax = plt.gca()

    patch_options = {'Line': patch_line, 'Polygon': patch_polygon, 'Point': patch_point}

    if layer_color is not None:
        try:
            color_patch = patch_options[gl.geom_type]()
        except ValueError:
            raise ValueError("'{}' is not a valid color".format(layer_color))
        handles.append(color_patch)

    # Use geopandas plotting function
    if gl.geom_type == "Point":
        ax = gl._gpd_df.plot(ax=ax, column=attribute, color=layer_color, marker=marker, label=layer_label, **kwargs)
    else:
        ax = gl._gpd_df.plot(ax=ax, column=attribute, color=layer_color, label=layer_label, **kwargs)

    if labels is not None:
        check_type(labels, str)
        # label_list = []
        new_layer = gl.shallow_copy()
        new_layer['label_point'] = new_layer.geometry.apply(lambda x: x.representative_point().coords[:])
        new_layer['label_point'] = [lb[0] for lb in new_layer["label_point"]]

        for _, row in new_layer.iterrows():
            xy = row["label_point"]
            s = row["name"]
            ax.annotate(s=s, xy=xy)
            # label_list.append(pyplot.text(xy[0], xy[1], s))

        # adjust_text(label_list, only_move={'text': 'xy'}, force_points=0.15)

    # This function return a new attribute to Axes object: handles
    ax.handles = handles

    return ax
