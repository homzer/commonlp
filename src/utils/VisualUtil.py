import os

import numpy
from PIL import Image


def draw_array_img(array, output_dir):
    """
    draw images of array
    :param output_dir: directory of images.
    :param array: 3-D array [batch, width, height]
    """
    def array2Picture(arr, name):
        img = Image.fromarray(arr * 255)
        img = img.convert('L')
        img.save(os.path.join(output_dir, "img-%d.jpg" % name))

    for i, mtx in enumerate(array):
        array2Picture(numpy.array(mtx), i)
