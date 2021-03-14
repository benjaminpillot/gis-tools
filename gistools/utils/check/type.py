# -*- coding: utf-8 -*-

""" Check type functions

Check type of variables, class attributes and return
error(s) if necessary
"""
import os
from inspect import signature
from functools import wraps
from collections.abc import Collection


# Decorator for asserting type of function input arguments
def type_assert(*ty_args, **ty_kwargs):
    """ Type asserting decorator

    For type, just define the required type.
    :param ty_args:
    :param ty_kwargs:
    :return:

    :Example:
        >>> @type_assert(int, int)
        >>> def add(x, y)
        >>>     return x + y
    """
    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def collection_type_assert(*ty_args, **ty_kwargs):
    """ Type asserting decorator for collections

    For collections, define a dictionary with 'length',
    'collection' and 'type' keys.
    :param ty_args:
    :param ty_kwargs:
    :return:

    :Example:
        >>> @collection_type_assert(x={'length':2, 'collection':tuple, 'type':float}, y=dict(length=3, collection=(
        >>> list,tuple), type=(float,int))
    """
    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]['collection']) or len(value) != bound_types[
                        name]['length'] or not all(isinstance(element, bound_types[name]['type']) for element in
                                                   value):
                            raise TypeError("Argument {} must be {}, have length {} and filled with {}".format(name,
                                            bound_types[name]['collection'], bound_types[name]['length'],
                                            bound_types[name]['type']))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def check_type(*args) -> None:
    """Check type of arguments

    :param args: tuple list of argument/type
    :return:

    :Example:
        >>> check_type(args[0], str, args[2], float)

        * May use tuple for checking against a list of valid types
        >>> check_type(args[0], (str, float, int))
    """
    if len(args) % 2 == 0:
        for item in range(0, len(args), 2):
            if not isinstance(args[item], args[item + 1]):
                raise TypeError("Type of argument {} is '{}' but must be '{}'".format(
                    item//2 + 1, type(args[item]).__name__, args[item + 1].__name__))


def check_type_in_collection(collection: Collection, _type, include: bool = True) -> None:
    """ Check type of elements in collection

    Check type of elements in collection through including or
    excluding specific types.

    :param collection: list/tuple/set/ndarray/etc. collection of data
    :type collection: Collection
    :param _type: type or tuple list of types
    :type _type: type/tuple
    :param include: should collection include or exclude type ?
    :type include: boolean
    :return: none

    :Example:
        >>> check_type_in_collection([2, 2.0, "string"], (int, float, str))
        OK

        * may check by excluding unwilling types
        >>> check_type_in_collection([2, 2.0, "string"], str, include=False)
        raise error
    """
    check_type(collection, Collection, _type, (type, tuple), include, bool)
    if include is True and not all(isinstance(element, _type) for element in collection):
        raise TypeError("All elements in collection '{}' must be {}".format(type(collection), _type))
    elif include is False and any(isinstance(element, _type) for element in collection):
        raise TypeError("Collection '{}' cannot contain {}".format(type(collection), _type))


def isfile(file):
    """ Check if input is a file

    Return True if input is a file, False otherwise
    :param file:
    :return:
    """
    try:
        return os.path.isfile(file)
    except (TypeError, ValueError):
        return False


def is_iterable(iterable):
    """ Check if input is iterable

    :param iterable:
    :return:
    """

    try:
        iter(iterable)
        return True
    except TypeError:
        return False


def is_property(obj, attribute):
    """ Check if object attribute is a property

    :param obj:
    :param attribute:
    :return:
    """
    try:
        return isinstance(type(obj).__getattribute__(obj, attribute), property)
    except AttributeError:
        return False
