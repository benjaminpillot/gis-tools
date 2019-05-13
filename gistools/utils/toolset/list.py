# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def split_list_by_index(alist, indices, include=False):
    """ Split a list by index ranges

    :param alist:
    :param indices:
    :param include: include index value as a "pivot" (end and start of broken sub-lists): True or False
    :return:
    """
    # Be sure there is no repetitive indices
    indices = list(set(indices))
    if include:
        indices = sorted([idx for idx in indices if idx < len(alist)])
        return [alist[i:j + 1] for i, j in zip([0] + indices, indices + [len(alist) - 1])]
    else:
        indices = sorted([idx for idx in indices if 0 < idx < len(alist)])
        return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]
