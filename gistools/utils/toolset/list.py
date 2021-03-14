# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

from math import ceil

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


def flatten(alist):
    """ Return a list of items from a list of list

    :param alist:
    :return:
    """
    return [item for sublist in alist for item in sublist]


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


def str_to_list(a_string):
    """ Convert string to list

    :param a_string:
    :return:
    """
    try:
        return a_string.splitlines()
    except AttributeError:
        return a_string


def chunks(alist, n):
    """ Return n chunks from list

    :param alist: a list
    :param n: number of returned chunks
    :return: a generator of chunks
    """
    chunk_size = ceil(len(alist)/n)
    for i in range(0, len(alist), chunk_size):
        yield alist[i:i+chunk_size]


def split_list_into_chunks(alist, n):
    """ Split a list into n chunks

    :param alist:
    :param n:
    :return:
    """
    return list(chunks(alist, n))
