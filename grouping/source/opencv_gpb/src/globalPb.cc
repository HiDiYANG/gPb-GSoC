//
//    cv_globalPb.cc
//    globalPb
//
//    Created by Di Yang on 31/05/13.
//    Copyright (c) 2013 The Australian National University. All rights reserved.
//

#include "globalPb.hh"
#include <assert.h>
#include <math.h>

using namespace std;
//using namespace cv;

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
	       cv::Mat & bg1)
  {
    double* weights;
    weigths = _mPb_Weights(image.channels());
    cv::Mat chR = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
    cv::Mat chG = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
    cv::Mat chB = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
    
    if(image.channels == 3)
      cv::split(image, chR, chG, chB);
    else{
      cv::clone(image, chR);
      cv::clone(image, chG);
      cv::clone(image, chB);
    }
  }
  

  void 
  globalPb(const cv::Mat & image,
	   cv:: Mat & gPb)
  {
    gPb = cv::Mat::zeros(image.rows, image.cols, CV_64FC3);
    double *weights;
    weights = _gPb_Weights(image.channels());
    for(size_t i = 0; i < 13; i++)
      cout<<"weight["<<i<<"]="<<weights[i]<<endl;

    //multiscalePb - mPb
  
    //spectralPb   - sPb

    //globalPb     - gPb
  
  }
}
