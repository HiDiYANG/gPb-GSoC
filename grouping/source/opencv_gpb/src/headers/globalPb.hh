#include <iostream>
#include <vector>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

namespace cv
{
  void 
  globalPb(const cv::Mat & image,
	   cv::Mat & gPb,
	   cv::Mat & gPb_thin,
	   vector<cv::Mat> & gPb_ori);

  void
  pb_parts_final_selected(vector<cv::Mat> & layers,
			  vector<vector<cv::Mat> > & gradients);
  
  void 
  MakeFilter(const int radii,
	     const double theta,
	     cv::Mat & kernel);
  
  void
  multiscalePb(const cv::Mat & image,
	       cv::Mat & mPb_max,
	       vector<vector<cv::Mat> > & gradients);   
}
