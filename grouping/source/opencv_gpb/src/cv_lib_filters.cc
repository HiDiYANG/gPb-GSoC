//
//    cv_lib_filters:
//       An extended library of opencv gaussian-based filters.
//       contains:
//       1D anistropic gaussian filters
//       2D anistropic gaussian filters
//       2D central-surrouding gaussian filters
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University. 
//    and Willow Garage inc.
//    All rights reserved.
//    
//

#include "cv_lib_filters.hh"

#define X_ORI 1
#define Y_ORI 0
#define ZERO 0
#define NON_ZERO 1
#define HILBRT_ON 1
#define HILBRT_OFF 0
#define PI 3.141592653

using namespace std;

namespace libFilters
{ 
  /********************************************************************************
   * Distribution Normalize and Mean value shifting
   ********************************************************************************/
  void
  normalizeDistr(const cv::Mat & input,
		 cv::Mat & output,
		 bool label)
  {
    bool flag = label ? ZERO : NON_ZERO;
    input.copyTo(output);
    output.convertTo(output, CV_32FC1);
    cv::Mat ones = cv::Mat::ones(output.rows, output.cols, output.type());
    double sumAbs = 0.0;
    double mean = 0.0;
    /* If required, zero-mean shift*/
    if(flag){
      for(size_t i=0; i<output.rows; i++)
	for(size_t j=0; j< output.cols; j++)
	  if(flag)
	    mean += output.at<float>(i,j);
      mean = mean/(double(output.rows*output.cols));
      cv::addWeighted(output, 1.0, ones, -mean, 0.0, output);
    }
    /* Distribution Normalized */
    for(size_t i=0; i<output.rows; i++)
      for(size_t j=0; j< output.cols; j++)
	sumAbs += fabs(output.at<float>(i,j));
    cv::divide(output, ones, output, 1.0/sumAbs);
  }

  /********************************************************************************
   * Matrix Rotation
   ********************************************************************************/
  void rotate_2D_crop(const cv::Mat & input,
		      cv::Mat & output,
		      double ori,
		      int len_cols,
		      int len_rows)
  {
    cv::Mat rotate_M = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::Mat tmp;
    cv::Point center = cv::Point((input.cols-1)/2, (input.rows-1)/2);
    double angle = ori/PI*180.0;
    rotate_M = cv::getRotationMatrix2D(center, angle, 1.0);
    /* Apply rotation transformation to a matrix */
    cv::warpAffine(input, tmp, rotate_M, input.size());
    
    /* Cropping */
    int border_rows = (input.rows - len_rows)/2;
    int border_cols = (input.cols - len_cols)/2;
    cv::Rect cROI(border_cols, border_rows, len_cols, len_rows);
    output = tmp(cROI);
  }

  void rotate_2D(const cv::Mat & input,
		 cv::Mat & output,
		 double ori)
  {
    rotate_2D_crop(input, output, ori, input.cols, input.rows);
  }

  /********************************************************************************
   * Filters Generation
   ********************************************************************************/
  
  int
  support_rotated(int x,
		  int y,
		  double ori,
		  bool label)
  {
    double sin_ori, cos_ori, mag0, mag1;
    bool flag = label ? X_ORI : Y_ORI;
    if(flag){
      cos_ori = double(x)*cos(ori);
      sin_ori = double(y)*sin(ori);
    }
    else{
      cos_ori = double(y)*cos(ori);
      sin_ori = double(x)*sin(ori);
    }
    mag0 = fabs(cos_ori - sin_ori);
    mag1 = fabs(cos_ori + sin_ori);
    return int(((mag0 > mag1)? mag0 : mag1)+1.0);
  }

  void 
  _gaussianFilter1D(int half_len,
		    double sigma,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & output)
  {
    bool flag = hlbrt? HILBRT_ON : HILBRT_OFF; 
    int len = 2*half_len+1;
    cv::Mat ones = cv::Mat::ones(len, 1, CV_32F);
    double sum_abs;
    output  = cv::getGaussianKernel(len, sigma, CV_32F);
    if(deriv == 1){
      for(int i=0; i<len; i++){
	output.at<float>(i) = output.at<float>(i)*double(half_len-i);
      }
    }
    else if(deriv == 2){
      for(int i=0; i<len; i++){
	double x = double(i-half_len);
	output.at<float>(i) = output.at<float>(i)*(x*x/sigma-1.0); 
      }
    }
    if(flag){}
    if(deriv > 0)
      normalizeDistr(output, output, ZERO);
    else
      normalizeDistr(output, output, NON_ZERO);
  }

  void 
  gaussianFilter1D(double sigma,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & output)
  {
    int half_len = int(sigma*3.0);
    _gaussianFilter1D(half_len, sigma, deriv, hlbrt, output);
  }

  void
  _gaussianFilter2D(int half_len,
		    double ori,
		    double sigma_x,
		    double sigma_y,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & output)
  {
    /* rotate support ROI */
    int len = 2*half_len+1;
    int half_len_rotate_x = support_rotated(half_len, half_len, ori, X_ORI);
    int half_len_rotate_y = support_rotated(half_len, half_len, ori, Y_ORI);
    int half_rotate_len = (half_len_rotate_x > half_len_rotate_y)? half_len_rotate_x : half_len_rotate_y;
    int len_rotate= 2*half_rotate_len+1;    
    cv::Mat output_x, output_y;

    /*   Conduct Compution */    
    _gaussianFilter1D(half_rotate_len, sigma_x, 0,     HILBRT_OFF, output_x);
    _gaussianFilter1D(half_rotate_len, sigma_y, deriv, HILBRT_OFF, output_y);
    output = output_x*output_y.t();
    rotate_2D_crop(output, output, ori, len, len);
    
    /*  Normalize  */
    if(deriv > 0)
      normalizeDistr(output, output, ZERO);
    else
      normalizeDistr(output, output, NON_ZERO);
  }


  void 
  gaussianFilter2D(double ori,
		   double sigma_x,
		   double sigma_y,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & output)
  {
    bool flag = hlbrt? HILBRT_ON : HILBRT_OFF; 
    /* actual size of kernel */
    int half_len_x = int(sigma_x*3.0);
    int half_len_y = int(sigma_y*3.0);
    int half_len = (half_len_x>half_len_y)? half_len_x : half_len_y;
    _gaussianFilter2D(half_len, ori, sigma_x, sigma_y, deriv, hlbrt, output);
  }

  void
  _gaussianFilter2D_cs(int half_len,
		       double ori,
		       double sigma_x,
		       double sigma_y,
		       double scale_factor,
		       cv::Mat & output)
  {
    double sigma_x_c = sigma_x/scale_factor;
    double sigma_y_c = sigma_y/scale_factor;
    cv::Mat output_cen, output_sur;
    _gaussianFilter2D(half_len, ori, sigma_x_c, sigma_y_c, 0, HILBRT_OFF, output_cen);
    _gaussianFilter2D(half_len, ori, sigma_x,   sigma_y,   0, HILBRT_OFF, output_sur);
    cv::addWeighted(output_sur, 1.0, output_cen, -1.0, 0.0, output);
    normalizeDistr(output, output, ZERO);
  }

  void
  gaussianFilter2D_cs(double ori,
		      double sigma_x,
		      double sigma_y,
		      double scale_factor,
		      cv::Mat & output)
  {
    int half_len_x = int(sigma_x*3.0);
    int half_len_y = int(sigma_y*3.0);
    int half_len = (half_len_x>half_len_y)? half_len_x : half_len_y;    
    _gaussianFilter2D_cs(half_len, ori, sigma_x, sigma_y, scale_factor, output);
  }
  
  void _texton_Filters(int n_ori,
		       double sigma)
  {
    
  }
}
