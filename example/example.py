import numpy as np
from skimage.io import imread
import ctypes
from gSLICrPy import cuda_slicr
from numpy.ctypeslib import ndpointer


def main():
    image = imread('./1.jpg')
    img_size_y, img_size_x = image.shape[0:2]
    image = image[:, :, ::-1].flatten().astype('uint8')

    result_sp = np.zeros((img_size_x, img_size_y), dtype=np.int)

    time = cuda_slicr(image, img_size_x=img_size_x, img_size_y=img_size_y, n_segs=1000,
                      spixel_size=16, coh_weight=0.6, n_iters=10, color_space=2, segment_color_space=2,
                      segment_by_size=True, enforce_connectivity=True, out_name='1', time=-1)
    print(time)
    pass


def main2():
    dll = ctypes.cdll.LoadLibrary('../build/libDEMO.so')

    t = dll.add(ctypes.c_int(19), ctypes.c_int(27))
    print("\n", t)
    pass


def main22():
    dll = ctypes.cdll.LoadLibrary('../build/libDEMO.so')
    dll.add2.restype = ctypes.c_float

    t = dll.add2(ctypes.c_float(19.0), ctypes.c_float(27.0))
    print("\n", t)
    pass


def main3():
    image = imread('./1.jpg')
    img_size_y, img_size_x = image.shape[0:2]
    image = image[:, :, ::-1].flatten().astype('uint8')
    image = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    out_name = '1'.encode('utf-8')

    n_segs = 1000
    spixel_size = 16
    coh_weight = 0.6
    n_iters = 10
    color_space = 2
    segment_color_space = 2
    segment_by_size = True
    enforce_connectivity = True

    dll = ctypes.cdll.LoadLibrary('../build/libDEMO.so')
    dll.CUDA_gSLICr.restype = ctypes.c_float

    pyarray = np.zeros((img_size_y * img_size_x, ), dtype="int32")
    carray = (ctypes.c_int * len(pyarray))(*pyarray)

    t = dll.CUDA_gSLICr(ctypes.POINTER(ctypes.c_uint8)(image),
                        ctypes.c_int(img_size_x), ctypes.c_int(img_size_y), ctypes.c_int(n_segs),
                        ctypes.c_int(spixel_size), ctypes.c_float(coh_weight), ctypes.c_int(n_iters),
                        ctypes.c_int(color_space), ctypes.c_int(segment_color_space),  ctypes.c_bool(segment_by_size),
                        ctypes.c_bool(enforce_connectivity), ctypes.c_char_p(out_name), carray)
    result = np.asarray(carray).reshape((img_size_y, img_size_x))
    print(result)
    print("\n", t)
    pass


if __name__ == '__main__':
    main3()
    pass
