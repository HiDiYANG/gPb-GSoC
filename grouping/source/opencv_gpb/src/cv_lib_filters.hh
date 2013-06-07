#include <iostream>
#include <vector>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#define X_ORI 1
#define Y_ORI 0
#define ZERO 0
#define NON_ZERO 1
#define HILBRT_ON 1
#define HILBRT_OFF 0

namespace libFilters
{
  void
  normalizeDistr(const cv::Mat & input,
		 cv::Mat & output,
		 bool label);

  /********************************************************************************
   * Matrix Rotation
   ********************************************************************************/
  
  void rotate_2D_crop(const cv::Mat & input,
		      cv::Mat & output,
		      double ori,
		      int len_cols,
		      int len_rows);

  void rotate_2D(const cv::Mat & input,
		 cv::Mat & output,
		 double ori);
  
  int
  support_rotated(int x,
		  int y,
		  double ori,
		  bool label);
 
  /********************************************************************************
   * Filters Generation
   ********************************************************************************/
  
  void 
  _gaussianFilter1D(int half_len,
		    double sigma,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & output);

  void 
  gaussianFilter1D(double sigma,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & output);

  void
  _gaussianFilter2D(int half_len,
		    double ori,
		    double sigma_x,
		    double sigma_y,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & output);

  void 
  gaussianFilter2D(double ori,
		   double sigma_x,
		   double sigma_y,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & output);


  void
  _gaussianFilter2D_cs(int half_len,
		       double ori,
		       double sigma_x,
		       double sigma_y,
		       double scale_factor,
		       cv::Mat & output);
  
  void
  gaussianFilter2D_cs(double ori,
		      double sigma_x,
		      double sigma_y,
		      double scale_factor,
		      cv::Mat & output);
  
  void _texton_Filters(int n_ori,
		       double sigma);
}
