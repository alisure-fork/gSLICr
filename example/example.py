from skimage.io import imread

from gSLICrPy import cuda_slicr


def main():
    image = imread('./1.jpg')
    img_size_y, img_size_x = image.shape[0:2]
    image = image[:, :, ::-1].flatten().astype('uint8')

    cuda_slicr(image, img_size_x=img_size_x, img_size_y=img_size_y, n_segs=1000,
               spixel_size=16, coh_weight=0.6, n_iters=10, color_space=2, segment_color_space=2,
               segment_by_size=True, enforce_connectivity=True, out_name='1')
    pass


if __name__ == '__main__':
    main()
    pass
