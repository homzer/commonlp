import os

import numpy
import matplotlib.pyplot as plt
from PIL import Image

__all__ = ['draw_array_img', 'draw_histogram']


def __clean_or_make_dir(output_dir):
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


def draw_histogram(count_dict: dict):
    """
    Draw histogram according to labels and counts.
    len(labels) must be equal to len(counts)
    :param count_dict: dictionary of labels and count.
    which's key is `str` and value is `int`.
    """
    labels = []
    counts = []
    for item in count_dict.items():
        labels.append(item[0])
        counts.append(item[1])
    plt.figure()
    plt.bar(labels, counts, width=0.7, align='center')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.ylim(0, 800)
    plt.show()


def draw_array_img(array, output_dir, normalizing=False):
    """
    draw images of array
    :param normalizing: whether to normalize.
    :param output_dir: directory of images.
    :param array: 3-D array [batch, width, height]
    """
    assert numpy.array(array).ndim == 3
    __clean_or_make_dir(output_dir)

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
        mtx = normalize(mtx) if normalizing else mtx
        array2Picture(numpy.array(mtx), index)
