//
//    cv_globalPb.cc
//    globalPb
//
//    Created by Di Yang on 31/05/13.
//    Copyright (c) 2013 The Australian National University. All rights reserved.
//

#include "globalPb.hh"

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
    float ab_min = -73.0, ab_max = 95.0;
    FILE* pFile_L, *pFile_a, *pFile_b;
    
    int tmp_len;
    int n_bg = 3;
    int n_cg = 3;
    int n_tg = 3;
    int r_bg[3] = { 3,  5, 10 };
    int r_cg[3] = { 5, 10, 20 };
    int r_tg[3] = { 5, 10, 20 };
    
    cv::Mat color, grey;
    
    cv::merge(layers, color);
    cv::copyMakeBorder(color, color, border, border, border, border, BORDER_REFLECT);
    cv::cvtColor(color, grey, CV_RGB2GRAY);
    
    tmp_sigma = double(num_bins)*bg_smooth_sigma;
    tmp_len = 2*int(3.0*tmp_sigma+0.5)+1;    
    cv::Mat bg_smooth_kernel  = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_64FC1);
    tmp_sigma = double(num_bins)*cg_smooth_sigma;
    tmp_len = 2*int(3.0*tmp_sigma+0.5)+1;
    cv::Mat cga_smooth_kernel = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_64FC1);
    cv::Mat cgb_smooth_kernel = cv::getGaussianKernel(tmp_len, tmp_sigma, CV_64FC1);
    
    color.convertTo(color, CV_32FC3);
    //TODO: normalize color first
    //ps. It is BGR not RGB
    cv::pow(color, 2.5, color);
    cv::cvtColor(color, color, CV_BGR2Lab);
    pFile_L = fopen("L.txt", "w+");
    pFile_a = fopen("a.txt", "w+");
    pFile_b = fopen("b.txt", "w+");
    cv::split(color, layers);
    for(size_t c=0; c<3; c++){
      //cv::minMaxLoc(layers[c], &min_elem, &max_elem);
      for(size_t i=0; i<layers[c].rows; i++){
	for(size_t j=0; j<layers[c].cols; j++){
	  if(c==0)
	    fprintf(pFile_L, "%0.2f ", layers[c].at<float>(i,j) );
	  if(c==1)
	    fprintf(pFile_a, "%0.2f ", layers[c].at<float>(i,j) );
	  if(c==2)
	    fprintf(pFile_b, "%0.2f ", layers[c].at<float>(i,j) );
	}
	if(c==0)
	    fprintf(pFile_L, "\n" );
	  if(c==1)
	    fprintf(pFile_a, "\n" );
	  if(c==2)
	    fprintf(pFile_b, "\n" );
      }
    }
    fclose(pFile_L);
    fclose(pFile_a);
    fclose(pFile_b);
    
    cv::merge(layers, color);
    color.convertTo(color, CV_8UC3);
    
    cv::imshow("border", color);
    cv::imshow("layers", layers[0]);

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
