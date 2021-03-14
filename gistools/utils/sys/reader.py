# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

import numpy as np
import os

from gistools.utils.check.value import check_file


def read_hdr(hdr_file: str):
    """ Read hdr header file for geospatial image

    :param hdr_file:
    :return:
    """
    def data_type():
        hdr_fields["data_type"] = np.dtype(data_type[int(value)])

    def samples():
        hdr_fields["x_size"] = int(value)

    def lines():
        hdr_fields["y_size"] = int(value)

    def map_info():
        # Pixel easting and northing: it seems envi header files store top-left pixel centre of the image...
        val = value[1:-1].replace(" ", "").split(",")
        hdr_fields["proj"] = val[0]
        hdr_fields["x_res"] = float(val[5])
        hdr_fields["y_res"] = float(val[6])
        hdr_fields["x_origin"] = float(val[3]) - hdr_fields["x_res"] / 2
        hdr_fields["y_origin"] = float(val[4]) + hdr_fields["y_res"] / 2

    check_file(hdr_file, ext='.hdr')
    options = {'data type': data_type, 'samples': samples, 'lines': lines, 'map info': map_info}
    hdr_fields = {}
    data_type = {1: np.byte, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 6: np.complex, 9: np.complex64,
                 12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64}

    with open(hdr_file, 'r') as file:
        for line in file.readlines():
            try:
                key, value = line.split(sep=" = ", maxsplit=1)  # Or re.split("\s*=\s*(.+)", line)[0:-1],
                # or re.search("([^\=]+)\s+\=\s*(.+)", line)
                if key in options.keys():
                    options[key]()
            except ValueError:
                pass

    return hdr_fields


def read_img(img_file: str):
    """ Read img file associated with hdr header files

    :param img_file:
    :return:
    """
    check_file(img_file, ext='.img')
    hdr_file = os.path.splitext(img_file)[0] + ".hdr"
    img_info = read_hdr(hdr_file)

    with open(img_file, 'r') as file:
        image = np.fromfile(file, img_info['data_type'])

    image = image.reshape((img_info["x_size"], img_info["y_size"]))

    return {"image": image, "attributes": img_info}
