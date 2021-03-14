# -*- coding: utf-8 -*-

""" Check value functions

Check and validate value of variables, class attributes and return
error(s) if necessary
"""
import os
from collections import Collection

from gistools.utils.check.type import check_type, isfile, type_assert


def check_string(string: str, list_of_strings: Collection) -> str:
    """ Check validity of and return string against list of valid strings

    :param string: searched string
    :type string: str
    :param list_of_strings: list/tuple/set of valid strings string is to be checked against
    :type list_of_strings: list, tuple, set
    :return: validate string from list of strings if match
    :rtype: str

    :Example:
        >>> check_string("well", ["Well Done", "no match", "etc."])
        "Well Done"
        >>> check_string("slow", ("slow", "slowdown"))
        ValueError: input match more than one valid value among ["slow", "slowdown"]
        >>> check_string("few", {"missed", "little"})
        ValueError: input must match one of those: ["missed", "little"]
    """
    check_type(string, str, list_of_strings, Collection)
    [check_type(x, str) for x in list_of_strings]

    output_string = []

    for item in list_of_strings:
        if item.lower().startswith(string.lower()):
            output_string.append(item)

    if len(output_string) == 1:
        return output_string[0]
    elif len(output_string) == 0:
        raise ValueError("input must match one of those: {}".format(list_of_strings))
    elif len(output_string) > 1:
        raise ValueError("input match more than one valid value among {}".format(list_of_strings))


def check_file(file_to_check, ext=None) -> None:
    """ Check validity of file

    :param file_to_check: file user want to check
    :param ext: extension(s) required for the file
    :return:
    """
    if isfile(file_to_check) is False:
        raise ValueError("{} is not a valid file".format(file_to_check))

    if ext is not None:
        if type(ext) == str:
            if os.path.splitext(file_to_check)[-1] != ext:
                raise ValueError("File {} must have the following extension: {}".format(file_to_check, ext))
        else:
            try:
                for file_ext in ext:
                    if os.path.splitext(file_to_check)[-1] == file_ext:
                        return None
                raise ValueError("File {} must have one of the following extensions: {}".format(file_to_check, ext))
            except TypeError:
                raise RuntimeError("Extension must be a string or a collection of strings")


@type_assert(sub_collection=Collection, collection=Collection)
def check_sub_collection_in_collection(sub_collection, collection):
    """ Check if elements of sub_collection are in collection

    :param sub_collection:
    :param collection:
    :return:
    """
    if any([element not in collection for element in sub_collection]):
        raise ValueError("Some element(s) of %s '%s' are not in %s '%s'" % (type(sub_collection), sub_collection,
                         type(collection), collection))


def check_value_in_range(value, range_min, range_max):
    """

    :param value:
    :param range_min:
    :param range_max:
    :return:
    """
    if value < range_min or value > range_max:
        raise ValueError("Value '%d' is outside range [%d, %d]" % (value, range_min, range_max))
