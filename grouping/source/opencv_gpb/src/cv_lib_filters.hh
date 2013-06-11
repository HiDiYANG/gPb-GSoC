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
#define SAME_SIZE 1
#define EXPAND_SIZE 0
#define OE_EVEN 1
#define OE_ODD 0

namespace libFilters
{
  /********************************************************************************
   * Hilbert Transform
   ********************************************************************************/
  void
  convolveDFT(const cv::Mat & inputA,
	      const cv::Mat & inputB,
	      cv::Mat & output);

  void
  hilbertTransform1D(const cv::Mat & input,
		     cv::Mat & output,
		     bool label);

  /********************************************************************************
   * Matrix Rotation
   ********************************************************************************/
  
  void 
  rotate_2D_crop(const cv::Mat & input,
		 cv::Mat & output,
		 double ori,
		 int len_cols,
		 int len_rows);

  void 
  rotate_2D(const cv::Mat & input,
	    cv::Mat & output,
	    double ori);
 
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
		       double sigma_x,
		       double sigma_y,
		       double scale_factor,
		       cv::Mat & output);
  
  void
  gaussianFilter2D_cs(double sigma_x,
		      double sigma_y,
		      double scale_factor,
		      cv::Mat & output);
  
  void
  oeFilters(int n_ori,
	    double sigma,
	    std::vector<cv::Mat> & filters,
	    bool label);
  
  void 
  textonFilters(int n_ori,
		double sigma,
		std::vector<cv::Mat> & filters);

  void
  texton(const cv::Mat & input,
	 std::vector<cv::Mat> & filtered,
	 int n_ori,
	 double sigma_sm,
	 double sigma_lg); 
}
