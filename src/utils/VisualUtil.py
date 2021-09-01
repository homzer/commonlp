import os

import numpy
from PIL import Image

__all__ = ['draw_array_img']


def clean_or_make_dir(output_dir):
    if os.path.exists(output_dir):
        def del_file(path):
            dir_list = os.listdir(path)
            for item in dir_list:
                item = os.path.join(path, item)
                del_file(item) if os.path.isdir(item) else os.remove(item)
        try:
            del_file(output_dir)
        except Exception as e:
            print(e)
            print('please remove the files of output dir.')
            exit(-1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def draw_array_img(array, output_dir):
    """
    draw images of array
    :param output_dir: directory of images.
    :param array: 3-D array [batch, width, height]
    """
    assert numpy.array(array).ndim == 3
    clean_or_make_dir(output_dir)

    def array2Picture(arr, name):
        img = Image.fromarray(arr * 255)
        img = img.convert('L')
        img.save(os.path.join(output_dir, "img-%d.jpg" % name))

    def normalize(arr):
        min_v = numpy.min(arr)
        max_v = numpy.max(arr)
        det_v = max_v - min_v
        for i, row in enumerate(arr):
            for j, x in enumerate(row):
                arr[i][j] = (x - min_v) / det_v
        return arr

    for index, mtx in enumerate(array):
        # mtx = normalize(mtx)
        array2Picture(numpy.array(mtx), index)
