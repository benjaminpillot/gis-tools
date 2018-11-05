# -*- coding: utf-8 -*-

""" System browsing functions

This module gathers all system-related utility functions,
such as file, directory search, etc.
"""

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2017, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'

import os
import re
from utils.check import check_type


def find_file(file: str, directory: str = os.getcwd(), sort=True) -> list:
    """ Find file(s) in directory

    :param file: file name/extension
    :param directory: relative/absolute path to directory
    :param sort: (bool) sort resulting list
    :return: list of file paths

    :Example:
        * Look for file extension in current directory
        >>> find_file(".py")
        ["/path/to/file.py", "/path/to/file2.py", ...]

        * Look for file name(s) in specific directory
        >>> find_file("filename", "/path/to/valid/directory")
        ["path/to/this-is-a-filename.txt", "path/to/filename.ods", ...]

        * Look for multiple file names/extension in current directory
        >>> list(set([item for item in find_file(...) for ... in ["filename", ".py"]]))

    """

    check_type(file, str, directory, str)  # Check input arguments

    result = []

    if file[0] == ".":  # pattern: file extension
        pattern = r'[^\\/:*?"<>|\r\n]+' + re.escape(file) + '$'
    else:  # pattern: file name (special regex characters are regarded as literal)
        pattern = re.escape(file)

    if os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for name in files:
                if re.search(pattern, name) is not None:
                    result.append(os.path.join(root, name))
    else:
        raise ValueError("'{}' is not a valid directory path".format(directory))

    if sort:
        result = sorted(result)

    return result
