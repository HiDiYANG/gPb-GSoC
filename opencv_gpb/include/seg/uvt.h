#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <queue>
#include <vector>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#ifndef __APPLE__
#include <values.h>
#else
#include <float.h>
#endif

#define DOUBLE_SIZE 1
#define SINGLE_SIZE 0

namespace cv {
void uvt(const cv::Mat & ucm_mtr,
         const cv::Mat & seeds,
         cv::Mat & boundary,
         cv::Mat & labels,
         bool sz);

void ucm2seg(const cv::Mat & ucm_mtr,
             cv::Mat & boundary,
             cv::Mat & labels,
             double thres,
             bool sz);

}
