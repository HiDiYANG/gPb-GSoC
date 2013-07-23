//
//    contour2ucm
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 22/07/13.
//    Copyright (c) 2013 The Australian National University. 
//    and Willow Garage inc.
//    All rights reserved.
//    
//

#include "contour2ucm.hh"

using namespace std;

namespace cv
{
  void creat_finest_partition(const cv::Mat & gPb,
			      cv::Mat & ws_wt)
  {
    cv::Mat temp = cv::Mat::zeros(gPb.rows, gPb.cols, CV_32FC1);
    cv::Mat ones = cv::Mat::ones(gPb.rows, gPb.cols, CV_32FC1);
    cv::multiply(gPb, ones, temp, 255.0);
    temp.convertTo(temp, CV_64FC1);
    cv::watershedFull(temp, 1, ws_wt);
  }

  void contour2ucm(const cv::Mat & gPb,
		   const vector<cv::Mat> & gPb_ori,
		   cv::Mat & ws_wt)
  {
    cout<<"here ... "<<endl;
    creat_finest_partition(gPb, ws_wt);
    cout<<"end ..."<<endl;
  }
}
