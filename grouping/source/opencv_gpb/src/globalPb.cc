//
//    cv_globalPb.cc
//    globalPb
//
//    Created by Di Yang on 31/05/13.
//    Copyright (c) 2013 The Australian National University. All rights reserved.
//

#include "globalPb.hh"
#define X_ORI 1
#define Y_ORI 0
#define ZERO 0
#define NON_ZERO 1
#define HILBRT_ON 1
#define HILBRT_OFF 0
#define PI 3.141592653

using namespace std;

namespace
{ static double* 
  _gPb_Weights(int nChannels)
  {
    double *weights = new double[13];
    if(nChannels == 3){
      weights[0] = 0.0;    weights[1] = 0.0;    weights[2] = 0.0039;
      weights[3] = 0.0050; weights[4] = 0.0058; weights[5] = 0.0069;
      weights[6] = 0.0040; weights[7] = 0.0044; weights[8] = 0.0049;
      weights[9] = 0.0024; weights[10]= 0.0027; weights[11]= 0.0170;
      weights[12]= 0.0074;
    }else{
      weights[0] = 0.0;    weights[1] = 0.0;    weights[2] = 0.0054;
      weights[3] = 0.0;    weights[4] = 0.0;    weights[5] = 0.0;
      weights[6] = 0.0;    weights[7] = 0.0;    weights[8] = 0.0;
      weights[9] = 0.0048; weights[10]= 0.0049; weights[11]= 0.0264;
      weights[12]= 0.0090;
    }
    return weights;
  }

  static double* 
  _mPb_Weights(int nChannels)
  {
    double *weights = new double[13];
    if(nChannels == 3){
      weights[0] = 0.0146; weights[1] = 0.0145; weights[2] = 0.0163;
      weights[3] = 0.0210; weights[4] = 0.0243; weights[5] = 0.0287;
      weights[6] = 0.0166; weights[7] = 0.0185; weights[8] = 0.0204;
      weights[9] = 0.0101; weights[10]= 0.0111; weights[11]= 0.0141;
    }else{
      weights[0] = 0.0245; weights[1] = 0.0220; weights[2] = 0.0;
      weights[3] = 0.0;    weights[4] = 0.0;    weights[5] = 0.0;
      weights[6] = 0.0;    weights[7] = 0.0;    weights[8] = 0.0;
      weights[9] = 0.0208; weights[10]= 0.0210; weights[11]= 0.0229;
    }
    return weights;
  }

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

namespace cv
{
  void
  pb_parts_final_selected(vector<cv::Mat> & layers
			  /*
			  cv::Mat & textons,
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
			  vector<cv::Mat> & tg_r20*/)
  {
    int n_ori      = 8;                       // number of orientations
    int num_bins = 25;                        // bins for L, b, a
    int border     = 30;                      // border pixels
    double bg_smooth_sigma = 0.1;             // bg histogram smoothing sigma
    double cg_smooth_sigma = 0.05;            // cg histogram smoothing sigma
    double sigma_tg_filt_sm = 2.0;            // sigma for small tg filters
    double sigma_tg_filt_lg = sqrt(2.0)*2.0;  // sigma for large tg filters
    
    int n_bg = 3;
    int n_cg = 3;
    int n_tg = 3;
    int r_bg[3] = { 3,  5, 10 };
    int r_cg[3] = { 5, 10, 20 };
    int r_tg[3] = { 5, 10, 20 };
    
    cv::Mat color, grey, ones, bg_smooth_kernel, cga_smooth_kernel, cgb_smooth_kernel;
    cv::merge(layers, color);
    cv::copyMakeBorder(color, color, border, border, border, border, BORDER_REFLECT);
    cv::cvtColor(color, grey, CV_BGR2GRAY);
    ones = cv::Mat::ones(color.rows, color.cols, CV_32FC1);
    
    // Histogram filter generation
    gaussianFilter1D(double(num_bins)*bg_smooth_sigma, 0, false, bg_smooth_kernel);
    gaussianFilter1D(double(num_bins)*cg_smooth_sigma, 0, false, cga_smooth_kernel);
    gaussianFilter1D(double(num_bins)*cg_smooth_sigma, 0, false, cgb_smooth_kernel);
    
    // Normalize color channels
    color.convertTo(color, CV_32FC3);
    cv::split(color, layers);
    for(size_t c=0; c<3; c++)
      cv::multiply(layers[c], ones, layers[c], 1.0/255.0);
    cv::merge(layers, color);
    
    // Color convert, including gamma correction
    cv::cvtColor(color, color, CV_BGR2Lab);

    // Normalize Lab channels
    cv::split(color, layers);
    for(size_t c=0; c<3; c++)
      for(size_t i=0; i<layers[c].rows; i++){
	for(size_t j=0; j<layers[c].cols; j++){
	  if(c==0)
	    layers[c].at<float>(i,j) = layers[c].at<float>(i,j)/100.0;
      	  else
	    layers[c].at<float>(i,j) = (layers[c].at<float>(i,j)+73.0)/168.0;
	  if(layers[c].at<float>(i,j) < 0.0)
	    layers[c].at<float>(i,j) = 0.0;
	  else if(layers[c].at<float>(i,j) > 1.0)
	    layers[c].at<float>(i,j) = 1.0;
	  float bin = floor(layers[c].at<float>(i,j)*float(num_bins));
	  if(bin == float(num_bins)) bin--;
	  layers[c].at<float>(i,j)=bin/24.0;
	}
      }
    cv::merge(layers, color);
    cv::imshow("quantized", color);
    
    /* Test rotated gaussian filter */
    cv::Mat g;
    //gaussianFilter2D(1.1781, 2.0, 2.0, 2, HLBRT_OFF, g);

    gaussianFilter2D_cs(0, 2.0, 2.0, M_SQRT2, g);
    FILE* pFile;
    pFile = fopen("gaussian_k.txt","w+");
    cout<<"writing into gaussian_k.txt ..."<<endl;
    for(size_t i=0; i<g.rows; i++){
      for(size_t j=0; j<g.cols; j++)
	fprintf(pFile,"%f ", g.at<float>(i,j));
      fprintf(pFile, "\n");
    }
    fclose(pFile);    
  }
  
  void 
  MakeFilter(const int radii,
	     const double theta,
	     cv::Mat & kernel)
  {
    double ra, rb, ira2, irb2;
    double sint, cost, ai, bi;
    double x[5] = {0};
    int wr;
    cv::Mat A = cv::Mat::zeros(3, 3, CV_32FC1);
    cv::Mat y = cv::Mat::zeros(3, 1, CV_32FC1);
    ra = MAX(1.5, double(radii));
    rb = MAX(1.5, double(radii)/4);
    ira2 = 1.0/(pow(ra, 2));
    irb2 = 1.0/(pow(rb, 2));
    wr = int(MAX(ra, rb));
    kernel = cv::Mat::zeros(2*wr+1, 2*wr+1, CV_32FC3);
  
    sint = sin(theta);
    cost = cos(theta);
    for(size_t i = 0; i <= 2*wr; i++)
      for(size_t j = 0; j <= 2*wr; j++){
	ai = -(double(i)-double(wr))*sint + (double(j)-double(wr))*cost;
	bi =  (double(i)-double(wr))*cost + (double(j)-double(wr))*sint;
	if((ai*ai*ira2 + bi*bi*irb2) > 1) continue;
	for(size_t n=0; n < 5; n++)
	  x[n] += pow(ai, double(n));
      }
    for(size_t i=0; i < 3; i++)
      for(size_t j = i; j < i+3; j++){
	A.at<float>(i, j-i) = x[j];
      }
    A = A.inv(DECOMP_SVD);
    for(size_t i = 0; i <= 2*wr; i++)
      for(size_t j = 0; j <= 2*wr; j++){
	ai = -(double(i)-double(wr))*sint + (double(j)-double(wr))*cost;
	bi =  (double(i)-double(wr))*cost + (double(j)-double(wr))*sint;
	if((ai*ai*ira2 + bi*bi*irb2) > 1) continue;
	for(size_t n=0; n < 3; n++)
	  y.at<float>(n,0) = pow(ai, double(n));
	y = A*y;
	for(size_t n=0; n < 3; n++)
	  kernel.at<Vec3d>(j,i)[n] = y.at<float>(n,0);
      }
  }

  void
  multiscalePb(const cv::Mat & image,
	       vector<cv::Mat> & layers)
  {
    double* weights;
    weights = _mPb_Weights(image.channels());
    layers.resize(3); 
    if(image.channels() == 3)
      cv::split(image, layers);
    else
      for(size_t i=0; i<3; i++)
	image.copyTo(layers[i]);
  }
  
  void 
  globalPb(const cv::Mat & image,
	   cv:: Mat & gPb)
  {
    gPb = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
    double *weights;
    weights = _gPb_Weights(image.channels());
    /*for(size_t i = 0; i < 13; i++)
      cout<<"weight["<<i<<"]="<<weights[i]<<endl;*/

    //multiscalePb - mPb
  
    //spectralPb   - sPb

    //globalPb     - gPb
  
  }
}
