// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>
#include <stdexcept>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define color_space_XYZ 0
#define color_space_LAB 1
#define color_space_RGB 2

using namespace std;
using namespace cv;


extern "C"
void load_image_from_Mat_to_UChar4(const Mat& inimg, gSLICr::UChar4Image* outimg){
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
	for (int y = 0; y < outimg->noDims.y;y++){
		for (int x = 0; x < outimg->noDims.x; x++){
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
	}
}


extern "C"
void load_image_from_UChar4_to_Mat(const gSLICr::UChar4Image* inimg, Mat& outimg){
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
	for (int y = 0; y < inimg->noDims.y; y++){
		for (int x = 0; x < inimg->noDims.x; x++){
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
	}
}


extern "C"
float CUDA_gSLICr(unsigned char* image, int img_size_x, int img_size_y, int n_segs, int spixel_size,
                  float coh_weight, int n_iters, int color_space, int segment_color_space,
                  bool segment_by_size, bool enforce_connectivity, const char* out_name, int data_ptr[]){
    StopWatchInterface *my_timer;
    sdkCreateTimer(&my_timer);

	// gSLICr settings
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = img_size_x;
	my_settings.img_size.y = img_size_y;
	my_settings.no_segs = n_segs;
	my_settings.spixel_size = spixel_size;
	my_settings.coh_weight = coh_weight;
	my_settings.no_iters = n_iters;
	my_settings.do_enforce_connectivity = enforce_connectivity; // whether or not run the enforce connectivity step

    // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	if(segment_color_space == color_space_XYZ)
		my_settings.color_space = gSLICr::XYZ;
	else if(segment_color_space == color_space_LAB)
		my_settings.color_space = gSLICr::CIELAB;
	else if(segment_color_space == color_space_RGB)
		my_settings.color_space = gSLICr::RGB;
	else
		throw std::invalid_argument("Assigned color space is not valid");

	if(segment_by_size)
		my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	else
		my_settings.seg_method = gSLICr::GIVEN_NUM;

    // Size
	Size s(my_settings.img_size.x, my_settings.img_size.y);

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	Mat boundry_draw_frame;
	boundry_draw_frame.create(s, CV_8UC3);

	Mat frame(s, CV_8UC3, (void *) image);

	load_image_from_Mat_to_UChar4(frame, in_img);

	sdkResetTimer(&my_timer);
	sdkStartTimer(&my_timer);

    gSLICr_engine->Process_Frame(in_img);

    const gSLICr::IntImage* idx_img = gSLICr_engine->Get_Seg_Res();
    const int* data_ptr2 = idx_img->GetData(MEMORYDEVICE_CPU);

    for (int i = 0; i < img_size_x * img_size_y; ++i){
        data_ptr[i] = data_ptr2[i];
    }

	sdkStopTimer(&my_timer);

    float one_time = sdkGetTimerValue(&my_timer);  // 返回时间
    return one_time;
}

int main(){
    return 875;
}