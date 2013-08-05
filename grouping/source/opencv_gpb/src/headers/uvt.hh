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

namespace cv{
  void uvt(const cv::Mat & ucm_mtr,
	   const cv::Mat & seeds,
	   cv::Mat & boundary,
	   cv::Mat & labels);
}
