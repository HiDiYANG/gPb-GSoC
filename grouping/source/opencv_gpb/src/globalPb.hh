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
	   cv::Mat & gPb);

  void 
  MakeFilter(const int radii,
	     const double theta,
	     cv::Mat & kernel);
  
  void
  multiscalePb(const cv::Mat & image,
	       std::vector<cv::Mat> & layers);
  
  void
  pb_parts_final_selected(vector<cv::Mat> & layers,
			  cv::Mat & texton,
			  vector<cv::Mat> & bg_r3,
			  vector<cv::Mat> & bg_r5,
			  vector<cv::Mat> & bg_r10,
			  vector<cv::Mat> & cga_r5,
			  vector<cv::Mat> & cga_r10,
			  vector<cv::Mat> & cga_r20,
			  vector<cv::Mat> & cgb_r5,
			  vector<cv::Mat> & cgb_r10,
			  vector<cv::Mat> & cgb_r20,
			  vector<cv::Mat> & tg_r5,
			  vector<cv::Mat> & tg_r10,
			  vector<cv::Mat> & tg_r20);
}

