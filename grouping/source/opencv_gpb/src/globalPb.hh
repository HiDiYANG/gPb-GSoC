#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>


namespace cv
{
  void 
  globalPb(const cv::Mat & image,
	   cv::Mat & gPb);

  void 
  MakeFilter(const int radii,
	     const double theta,
	     cv::Mat & kernel);
}

