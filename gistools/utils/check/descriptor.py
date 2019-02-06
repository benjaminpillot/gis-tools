# -*- coding: utf-8 -*-

""" Descriptor classes

Use descriptor classes to implement data models
"""

# __all__ = []
# __version__ = '0.1'
__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2017, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'

# import
from collections.abc import Collection


def protected_property(name):
    """ Protected property

    Function used for implementing
    repetitive "SetAccess = protected"
    of class properties
    :param name:
    :return:
    """
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)

    return prop


# Metaclass for simplifying descriptor attribute setter
class CheckedMeta(type):
    def __new__(mcs, class_name, bases, methods):
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(mcs, class_name, bases, methods)


# Base class
class Descriptor:
    def __init__(self, name=None, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


# Descriptor for enforcing type
class Typed(Descriptor):
    expected_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError("Type of attribute '{}' must be {} but is {}".format(self.name, self.expected_type,
                                                                                 type(value)))
        super().__set__(instance, value)


# Descriptor for enforcing types in collection
class TypedInCollection(Descriptor):
    expected_type_in_collection = type(None)

    def __set__(self, instance, value):
        if not all(isinstance(element, self.expected_type_in_collection) for element in value):
            raise TypeError("All elements in collection '{}' must be {}".format(self.name,
                                                                                self.expected_type_in_collection))
        super().__set__(instance, value)


# Descriptor for enforcing exclusion of types in collection
class TypedOutOfCollection(Descriptor):
    type_out_of_collection = type(None)

    def __set__(self, instance, value):
        if any(isinstance(element, self.type_out_of_collection) for element in value):
            raise TypeError("Collection '{}' cannot contain {}".format(self.name, self.type_out_of_collection))
        super().__set__(instance, value)


class UpperBoundInCollection(Descriptor):
    max = None

    def __init__(self, name=None, **kwargs):
        if "max" not in kwargs:
            raise TypeError("missing 'max' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if any(element > self.max for element in value):
            raise ValueError("Collection '{}' cannot contain value(s) > to {}".format(self.name,
                                                                                      self.max))
        super().__set__(instance, value)


class LowerBoundInCollection(Descriptor):
    min = None

    def __init__(self, name=None, **kwargs):
        if "min" not in kwargs:
            raise TypeError("missing 'min' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if any(element < self.min for element in value):
            raise ValueError("Collection '{}' cannot contain value(s) < to {}".format(self.name,
                                                                                      self.min))
        super().__set__(instance, value)


class BoundsInCollection(LowerBoundInCollection, UpperBoundInCollection):
    pass


class RangeInCollection(Descriptor):
    range = None

    def __init__(self, name=None, **kwargs):
        if "range" not in kwargs:
            raise TypeError("missing 'range' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if any(element not in self.range for element in value):
            raise ValueError("Value(s) in collection must match one of those: {}".format(self.name, self.range))
        super().__set__(instance, value)


# Descriptor for enforcing value from a list/set of values
class Range(Descriptor):
    range = None

    def __init__(self, name=None, **kwargs):
        if "range" not in kwargs:
            raise TypeError("missing 'range' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if value not in self.range:
            raise ValueError("Attribute '{}' must match one of those values: {}".format(self.name, self.range))
        super().__set__(instance, value)


class Length(Descriptor):
    length = None

    def __init__(self, name=None, **kwargs):
        if "length" not in kwargs:
            raise TypeError("missing 'length' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if len(value) != self.length:
            raise ValueError("'{}' must have len = {}".format(self.name, self.length))
        super().__set__(instance, value)


# Descriptor for enforcing lower bound on values
class LowerBound(Descriptor):
    min = None

    def __init__(self, name=None, **kwargs):
        if "min" not in kwargs:
            raise TypeError("missing 'min' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if value < self.min:
            raise ValueError("Attribute '{}' must be >= {}".format(self.name, self.min))
        super().__set__(instance, value)


# Descriptor for enforcing upper bound on values
class UpperBound(Descriptor):
    max = None

    def __init__(self, name=None, **kwargs):
        if "max" not in kwargs:
            raise TypeError("missing 'max' keyword argument")
        super().__init__(name, **kwargs)

    def __set__(self, instance, value):
        if value > self.max:
            raise ValueError("Attribute '{}' must be <= {}".format(self.name, self.max))
        super().__set__(instance, value)


# Descriptor for enforcing valid datetime values
# class Datetime(Descriptor):
#
#     def __init__(self, name=None):
#         super(Datetime, self).__init__(name)
#
#     def __set__(self, instance, value):


# Descriptor for enforcing non-negative values
class Unsigned(LowerBound):
    def __init__(self, name=None):
        super().__init__(name, min=0)


class Bounds(LowerBound, UpperBound):
    pass


class Integer(Typed):
    expected_type = int


class UnsignedInteger(Integer, Unsigned):
    pass


class Float(Typed):
    expected_type = float


class UnsignedFloat(Float, Unsigned):
    pass


class BoundedFloat(Float, Bounds):
    pass


class RangeFloat(Float, Range):
    pass


class CollectionOfData(Typed):
    expected_type = Collection


class TupleOfData(Typed):
    expected_type = tuple


class CollectionOfFloats(CollectionOfData, TypedInCollection):
    expected_type_in_collection = float


class CollectionOfInts(CollectionOfData, TypedInCollection):
    expected_type_in_collection = int


class CollectionOfBoundedFloats(CollectionOfFloats, BoundsInCollection):
    pass


class CollectionOfRangeFloats(CollectionOfFloats, RangeInCollection):
    pass


class TupleOfNumeric(CollectionOfFloats, CollectionOfInts, TupleOfData):
    pass


class PairTuple(TupleOfData, Length):
    length = 2


class PairTupleOfNumeric(TupleOfNumeric, PairTuple):
    pass


class String(Typed):
    expected_type = str


if __name__ == '__main__':
    pass
