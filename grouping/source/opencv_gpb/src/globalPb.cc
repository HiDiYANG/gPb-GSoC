//
//    globalPb
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University. 
//    and Willow Garage inc.
//    All rights reserved.
//    
//

#include "cv_lib_filters.hh"
#include "globalPb.hh"
#include <string>
#include <sstream>

using namespace std;
using namespace libFilters;

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
			  vector<cv::Mat> & tg_r20)
  {
    int n_ori      = 8;                       // number of orientations
    int num_bins   = 25;                      // bins for L, b, a
    int Kmean_bins = 64;                      // bins for texton
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
    
    cv::Mat color, grey, ones;
    cv::Mat bg_smooth_kernel, cg_smooth_kernel;
    cv::merge(layers, color);
    cv::cvtColor(color, grey, CV_BGR2GRAY);
    ones = cv::Mat::ones(color.rows, color.cols, CV_32FC1);
    
    // Histogram filter generation
    gaussianFilter1D(double(num_bins)*bg_smooth_sigma, 0, false, bg_smooth_kernel);
    gaussianFilter1D(double(num_bins)*cg_smooth_sigma, 0, false, cg_smooth_kernel);
    
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

	  //quantize color channels
	  
	  float bin = floor(layers[c].at<float>(i,j)*float(num_bins));
	  if(bin == float(num_bins)) bin--;
	  layers[c].at<float>(i,j)=bin;
	}
      }

    /********* END OF FILTERS INTIALIZATION ***************/

    cout<<"computing texton ... "<<endl;
    textonRun(grey, texton, n_ori, Kmean_bins, sigma_tg_filt_sm, sigma_tg_filt_lg);

    // L Channel
    cout<<"computing bg's ... "<<endl;
    gradient_hist_2D(layers[0], r_bg[0], n_ori, num_bins, bg_smooth_kernel, bg_r3);
    gradient_hist_2D(layers[0], r_bg[1], n_ori, num_bins, bg_smooth_kernel, bg_r5);
    gradient_hist_2D(layers[0], r_bg[2], n_ori, num_bins, bg_smooth_kernel, bg_r10);

    // a Channel
    cout<<"computing cga's ... "<<endl;
    gradient_hist_2D(layers[1], r_cg[0], n_ori, num_bins, cg_smooth_kernel, cga_r5);
    gradient_hist_2D(layers[1], r_cg[1], n_ori, num_bins, cg_smooth_kernel, cga_r10);
    gradient_hist_2D(layers[1], r_cg[2], n_ori, num_bins, cg_smooth_kernel, cga_r20);

    // b Channel
    cout<<"computing cgb's ... "<<endl;
    gradient_hist_2D(layers[2], r_cg[0], n_ori, num_bins, cg_smooth_kernel, cgb_r5);
    gradient_hist_2D(layers[2], r_cg[1], n_ori, num_bins, cg_smooth_kernel, cgb_r10);
    gradient_hist_2D(layers[2], r_cg[2], n_ori, num_bins, cg_smooth_kernel, cgb_r20);

    // T channel
    cout<<"computing tg's ... "<<endl;
    gradient_hist_2D(texton, r_tg[0], n_ori, Kmean_bins, tg_r5);
    gradient_hist_2D(texton, r_tg[1], n_ori, Kmean_bins, tg_r10);
    gradient_hist_2D(texton, r_tg[2], n_ori, Kmean_bins, tg_r20);
    
    texton.convertTo(texton, CV_8UC1);
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
    kernel = cv::Mat::zeros(2*wr+1, 2*wr+1, CV_32FC1);
  
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
	kernel.at<float>(j,i) = y.at<float>(0,0);
      }
  }

  void
  multiscalePb(const cv::Mat & image,
	       cv::Mat & mPb_max,
	       vector<cv::Mat> & mPb_all,
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
	       vector<cv::Mat> & tg_r20)
  {
    cv::Mat texton, kernel, index;
    vector<cv::Mat> layers;
    int n_ori = 8;
    int radii[4] ={3, 5, 10, 20};
    double* weights, *ori;
    weights = _mPb_Weights(image.channels());
    layers.resize(3); 
    if(image.channels() == 3)
      cv::split(image, layers);
    else
      for(size_t i=0; i<3; i++)
	image.copyTo(layers[i]);
    
    pb_parts_final_selected(layers, texton, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20);
    
    mPb_all.resize(n_ori);
    cout<<"Cues smoothing ..."<<endl;
    ori = standard_filter_orientations(n_ori, RAD);
    for(size_t idx=0; idx<n_ori; idx++){
      // radian 3
      MakeFilter(radii[0], ori[idx], kernel);
      cv::filter2D(bg_r3[idx],   bg_r3[idx],   CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      
      // radian 5
      MakeFilter(radii[1], ori[idx], kernel);
      cv::filter2D(bg_r5[idx],   bg_r5[idx],   CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(cga_r5[idx],  cga_r5[idx],  CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(cgb_r5[idx],  cgb_r5[idx],  CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(tg_r5[idx],   tg_r5[idx],   CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);

      // radian 10
      MakeFilter(radii[2], ori[idx], kernel);
      cv::filter2D(bg_r10[idx],  bg_r10[idx],  CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(cga_r10[idx], cga_r10[idx], CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(cgb_r10[idx], cgb_r10[idx], CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(tg_r10[idx],  tg_r10[idx],  CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);

      // radian 20
      MakeFilter(radii[3], ori[idx], kernel);
      cv::filter2D(cga_r20[idx], cga_r20[idx], CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(cgb_r20[idx], cgb_r20[idx], CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      cv::filter2D(tg_r20[idx],  tg_r20[idx],  CV_32F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
      
      mPb_all[idx] = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
      addWeighted(mPb_all[idx], 1.0, bg_r3[idx],   weights[0], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, bg_r5[idx],   weights[1], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, bg_r10[idx],  weights[2], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cga_r5[idx],  weights[3], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cga_r10[idx], weights[4], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cga_r20[idx], weights[5], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cgb_r5[idx],  weights[6], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cgb_r10[idx], weights[7], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, cgb_r20[idx], weights[8], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, tg_r5[idx],   weights[9], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, tg_r10[idx],  weights[10], 0.0, mPb_all[idx]);
      addWeighted(mPb_all[idx], 1.0, tg_r20[idx],  weights[11], 0.0, mPb_all[idx]);
      
      if(idx == 0){
	index = cv::Mat::zeros(image.rows, image.cols, CV_32SC1);
	mPb_all[idx].copyTo(mPb_max);
      }
      else
	for(size_t i=0; i<image.rows; i++)
	  for(size_t j=0; j<image.cols; j++)
	    if(mPb_max.at<float>(i,j) < mPb_all[idx].at<float>(i,j)){
	      index.at<int>(i,j) = idx;
	      mPb_max.at<float>(i,j) = mPb_all[idx].at<float>(i,j);
	    }
    }
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
