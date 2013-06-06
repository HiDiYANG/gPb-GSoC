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

  void
  normalizeDistr(const cv::Mat & input,
		 cv::Mat & output)
  {
    input.copyTo(output);
    output.convertTo(output, CV_32FC1);
    cv::Mat ones = cv::Mat::ones(output.rows, output.cols, output.type());
    double sumAbs = 0.0;
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
		      int len_x,
		      int len_y)
  {
    cv::Mat rotate_M = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::Point center = cv::Point((input.cols-1)/2, (input.rows-1)/2);
    cv::Size size(len_x, len_y);
    double angle = ori/PI*180.0;
    rotate_M = cv::getRotationMatrix2D(center, angle, 1.0);
    /* Apply rotation transformation to a matrix */
    cv::warpAffine(input, output, rotate_M, size);
    
    /* Cropping */
    

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
		    cv::Mat & gaussian)
  {
    int len = 2*half_len+1;
    cv::Mat ones = cv::Mat::ones(len, 1, CV_32F);
    double sum_abs;
    gaussian  = cv::getGaussianKernel(len, sigma, CV_32F);
    if(deriv == 1){
      for(int i=0; i<len; i++){
	gaussian.at<float>(i) = gaussian.at<float>(i)*double(half_len-i);
      }
    }
    else if(deriv == 2){
      for(int i=0; i<len; i++){
	double x = double(i-half_len);
	gaussian.at<float>(i) = gaussian.at<float>(i)*(x*x/sigma-1.0); 
      }
    }
    if(hlbrt){}
    normalizeDistr(gaussian, gaussian);
  }

  void 
  gaussianFilter1D(double sigma,
		   int deriv,
		   bool hlbrt,
		   cv::Mat & gaussian)
  {
    int half_len = int(sigma*3.0);
    _gaussianFilter1D(half_len, sigma, deriv, hlbrt, gaussian);
  }

  void 
  _gaussianFilter2D(double ori,
		    double sigma_x,
		    double sigma_y,
		    int deriv,
		    bool hlbrt,
		    cv::Mat & gaussian)
  {
    int half_len_x = int(sigma_x*3.0);
    int half_len_y = int(sigma_y*3.0);
    int half_len = (half_len_x>half_len_y)? half_len_x : half_len_y;
    int len = 2*half_len+1;
    
    int half_len_rotate_x = support_rotated(half_len, half_len, ori, X_ORI);
    int half_len_rotate_y = support_rotated(half_len, half_len, ori, Y_ORI);
    int half_rotate_len = (half_len_rotate_x > half_len_rotate_y)? half_len_rotate_x : half_len_rotate_y;
    int len_rotate= 2*half_rotate_len+1;
    cv::Mat gaussian_x, gaussian_y;
    
    /*   Conduct Compution */    
    _gaussianFilter1D(half_rotate_len, sigma_x, 0,     false, gaussian_x);
    _gaussianFilter1D(half_rotate_len, sigma_y, deriv, false, gaussian_y);
    
    /* debuging patch */
    /*FILE* pFile1, *pFile2;
    pFile1 = fopen("p1.txt","w+");
    pFile2 = fopen("p2.txt","w+");
    for(size_t i =0; i<gaussian_x.rows; i++){
      fprintf(pFile1, "%f\n", gaussian_x.at<float>(i,0));
      fprintf(pFile2, "%f\n", gaussian_y.at<float>(i,0));
    }
    fclose(pFile1);
    fclose(pFile2);*/

    cv::transpose(gaussian_y, gaussian_y);
    gaussian = gaussian_x*gaussian_y;
    rotate_2D_crop(gaussian, gaussian, ori, len, len);
    
    /*  Normalize  */
    normalizeDistr(gaussian, gaussian);
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
    double tmp_sigma;
    
    int tmp_len;
    int n_bg = 3;
    int n_cg = 3;
    int n_tg = 3;
    int r_bg[3] = { 3,  5, 10 };
    int r_cg[3] = { 5, 10, 20 };
    int r_tg[3] = { 5, 10, 20 };
    
    cv::Mat color, grey, ones;
    
    cv::merge(layers, color);
    cv::copyMakeBorder(color, color, border, border, border, border, BORDER_REFLECT);
    cv::cvtColor(color, grey, CV_BGR2GRAY);
    ones = cv::Mat::ones(color.rows, color.cols, CV_32FC1);
    
    tmp_sigma = double(num_bins)*bg_smooth_sigma;
    tmp_len = 2*int(3.0*tmp_sigma+0.5)+1;    
    cv::Mat bg_smooth_kernel  = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_32FC1);
    tmp_sigma = double(num_bins)*cg_smooth_sigma;
    tmp_len = 2*int(3.0*tmp_sigma+0.5)+1;
    cv::Mat cga_smooth_kernel = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_32FC1);
    cv::Mat cgb_smooth_kernel = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_32FC1);

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

    cv::Mat g;
    _gaussianFilter2D(1.5708, 2.5, 1.0, 2, false, g);    
    FILE* pFile;
    pFile = fopen("gaussian_k.txt","w+");
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
    cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat y = cv::Mat::zeros(3, 1, CV_64FC1);
    ra = MAX(1.5, double(radii));
    rb = MAX(1.5, double(radii)/4);
    ira2 = 1.0/(pow(ra, 2));
    irb2 = 1.0/(pow(rb, 2));
    wr = int(MAX(ra, rb));
    kernel = cv::Mat::zeros(2*wr+1, 2*wr+1, CV_64FC3);
  
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
	A.at<double>(i, j-i) = x[j];
      }
    A = A.inv(DECOMP_SVD);
    for(size_t i = 0; i <= 2*wr; i++)
      for(size_t j = 0; j <= 2*wr; j++){
	ai = -(double(i)-double(wr))*sint + (double(j)-double(wr))*cost;
	bi =  (double(i)-double(wr))*cost + (double(j)-double(wr))*sint;
	if((ai*ai*ira2 + bi*bi*irb2) > 1) continue;
	for(size_t n=0; n < 3; n++)
	  y.at<double>(n,0) = pow(ai, double(n));
	y = A*y;
	for(size_t n=0; n < 3; n++)
	  kernel.at<Vec3d>(j,i)[n] = y.at<double>(n,0);
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
    gPb = cv::Mat::zeros(image.rows, image.cols, CV_64FC3);
    double *weights;
    weights = _gPb_Weights(image.channels());
    /*for(size_t i = 0; i < 13; i++)
      cout<<"weight["<<i<<"]="<<weights[i]<<endl;*/

    //multiscalePb - mPb
  
    //spectralPb   - sPb

    //globalPb     - gPb
  
  }
}
