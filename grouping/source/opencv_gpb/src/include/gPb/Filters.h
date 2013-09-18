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
#define RAD 1
#define DEG 0

namespace cv
{
  /********************************************************************************
   * Hilbert Transform
   ********************************************************************************/
  void
  convolveDFT(const cv::Mat & inputA,
	      const cv::Mat & inputB,
	      cv::Mat & output,
	      bool label);

  void
  hilbertTransform1D(const cv::Mat & input,
		     cv::Mat & output,
		     bool label);
  
  //------------------------------------------
  double* 
  standard_filter_orientations(int n_ori,
			       bool label);


  /********************************************************************************
   * Matrix Rotation
   ********************************************************************************/
  
  void 
  rotate_2D_crop(const cv::Mat & input,
		 cv::Mat & output,
		 double ori,
		 int len_cols,
		 int len_rows,
		 bool label);

  void 
  rotate_2D(const cv::Mat & input,
	    cv::Mat & output,
	    double ori,
	    bool label);
 
  /********************************************************************************
   * Filters Generation
   ********************************************************************************/
  
  void 
  gaussianFilter1D(int half_len,
		    double sigma,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & output);

  void 
  gaussianFilter1D(double sigma,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & output);
  
 //-----------------------------------------------
  void
  gaussianFilter2D(int half_len,
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

  //-----------------------------------------------
  void
  gaussianFilter2D_cs(int half_len,
		       double sigma_x,
		       double sigma_y,
		       double scale_factor,
		       cv::Mat & output);
  
  void
  gaussianFilter2D_cs(double sigma_x,
		      double sigma_y,
		      double scale_factor,
		      cv::Mat & output);
  
 //-----------------------------------------------
  void
  gaussianFilters(int n_ori,
		  double sigma,
		  int deriv,
		  bool hlbrt,
		  double enlongation,
		  std::vector<cv::Mat> & filters);
  
  void
  oeFilters(int n_ori,
	    double sigma,
	    std::vector<cv::Mat> & filters,
	    bool label);
  
  //-----------------------------------------------
  void 
  textonFilters(int n_ori,
		double sigma,
		std::vector<cv::Mat> & filters);

  void
  textonRun(const cv::Mat & input,
	    cv::Mat & output,
	    int n_ori,
	    int Kmean_num,
	    double sigma_sm,
	    double sigma_lg); 

  cv::Mat 
  orientation_slice_map(int r, 
			int n_ori);

  //-----------------------------------------------
  void
  gradient_hist_2D(const cv::Mat & label,
		   int r,
		   int n_ori,
		   int num_bins,
		   cv::Mat & gaussian_kernel,
		   std::vector<cv::Mat> & gradients);

  void
  gradient_hist_2D(const cv::Mat & label,
		   int r,
		   int n_ori,
		   int num_bins,
		   std::vector<cv::Mat> & gradients);

  //-----------------------------------------------
  void 
  parallel_for_gradient_hist_2D(const cv::Mat & label,
				int r,
				int n_ori,
				int num_bins,
				cv::Mat & gaussian_kernel,
				std::vector<cv::Mat> & gradients);
}
