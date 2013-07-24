#include <iostream>
#include <vector>
#include <math.h>
#include <stack>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include "watershed.h"

namespace cv
{
  void creat_finest_partition(const cv::Mat & gPb,
			      cv::Mat & ws_wt);

  void contour2ucm(const cv::Mat & gPb,
		   const vector<cv::Mat> & gPb_ori,
		   cv::Mat & ws_wt);
 
}
