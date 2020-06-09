import ctypes
from ctypes import POINTER


def _gpu_slic(path_to_shared='../build/libDEMO.so'):
    dll = ctypes.CDLL(path_to_shared, mode=ctypes.RTLD_GLOBAL)
    func = dll.CUDA_gSLICr
    """
    int* CUDA_gSLICr(unsigned char* image, int img_size_x, int img_size_y, int n_segs,
                     int spixel_size, float coh_weight, int n_iters, int color_space,
                     int segment_color_space, bool segment_by_size, bool enforce_connectivity, char* out_name)
    """
    func.argtypes = [POINTER(ctypes.c_uint8),
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_char_p]
    return func


def cuda_slicr(image, img_size_x, img_size_y, n_segs, spixel_size, coh_weight, n_iters,
               color_space, segment_color_space, segment_by_size, enforce_connectivity, out_name):
    image = image.ctypes.data_as(POINTER(ctypes.c_uint8))
    out_name = out_name.encode('utf-8')

    return _gpu_slic()(image, img_size_x, img_size_y, n_segs, spixel_size, coh_weight, n_iters,
                       color_space, segment_color_space, segment_by_size, enforce_connectivity, out_name)
