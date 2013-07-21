#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#define MAX_ITER 10

/* 
 * This is the power iteration method to find the maximum
 * eigenvalue/eigenvector a n-by-n matrix. This method doesn't
 * require the matrix to be Hermitian for the maximum eigenvalue/eigenvecor.
 * But it DOES require the matrix to be Hermitian for the minimum
 * eigenvalue/vector. This approximation method may be improved by
 * setting a tolerance (currently the iteration is controlled by the number
 * of iterations, MAX).
 * 
 * Example: c = [1 0.5 0.2;0.5 1 0.5; 0.2 0.5 1];
 * then [u,v] = mPowerEig(c,0) is to find the largest eigenvalue/vector 
 * and  [u,v] = mPowerEig(c,1) is to find the minimum eigenvalue/vector
 * 
 * Reference: G.H. Golub, C.F. Van Load, "Matrix Computation"
 */

double power(const cv::Mat & a,
	     cv::Mat & q,
	     cv::Mat & z)
{
    double norm, lambda;
    int n = q.rows;
    z = cv::Mat::zeros(n, 1, CV_32FC1);
    
    for(size_t i = 0; i<MAX_ITER; i++){
      z=a*q;
      norm = 0.0;
      for(size_t j=0; j<n; j++){
	norm += z.at<float>(j, 0)*z.at<float>(j, 0);
	std::cout<<"z["<<j<<"] = "<<z.at<float>(j,0)<<std::endl;
      }
      norm = sqrt(norm);
      std::cout<<"norm = "<<norm<<std::endl;
      for(size_t j=0; j<n; j++){
	q.at<float>(j, 0)=z.at<float>(j, 0)/norm;
	std::cout<<"q["<<j<<"] = "<<q.at<float>(j,0)<<std::endl;
      }
      lambda = 0.0;
      for(size_t j=0; j<n; j++)
	lambda += q.at<float>(j, 0)*z.at<float>(j, 0);
    }
    return lambda;
}

double powermethod(const cv::Mat & x, 
		   int mode, 
		   cv::Mat & y)
{
  int n = x.rows;
  cv::Mat a = cv::Mat::zeros(x.rows, x.cols, CV_32FC1);
  cv::Mat b = cv::Mat::zeros(x.rows, x.cols, CV_32FC1);
  cv::Mat q = cv::Mat::zeros(x.rows, 1, CV_32FC1);
  cv::Mat ones_sq = cv::Mat::ones(x.rows, x.cols, CV_32FC1);
  cv::Mat ones = cv::Mat::ones(x.rows, 1, CV_32FC1);
  cv::Mat z;
  double lambda, lambdamin, w;
  x.copyTo(a);
 
  q.at<float>(0,0) = 1.0;
  lambda = power(a,q,z);
    
  
  /* mode ==0 is for maximum eigenvalue/vector */
  if (mode == 0) {
    q.copyTo(y);
    w = lambda;
  }
  /* else mode ~= 0, for minimum eigenvalue/vector */
  else {
    cv::multiply(a, ones_sq, b, -1.0);
    for (size_t i=0; i<n; i++) 
      b.at<float>(i,i) = lambda-a.at<float>(i,i);
    
    /* perform one more time power iteration method */
    q = cv::Mat::zeros(x.rows,1,CV_32FC1);
    q.at<float>(0,0)=1.0;
            
    /* iterative method to find maximum eigenvalue */
    lambdamin = power(b,q,z);
    
    cv::multiply(q, ones, y, -1.0);
    w = lambda - lambdamin;
  } 
  return w;
}

int main(int argc, char** agrv)
{
  cv::Mat c = cv::Mat::zeros(3, 3, CV_32FC1);
  c.at<float>(0, 0) = 2.0;
  c.at<float>(0, 1) = 16.0;
  c.at<float>(0, 2) = 8.0;
  c.at<float>(1, 0) = 4.0;
  c.at<float>(1, 1) = 14.0;
  c.at<float>(1, 2) = 8.0;
  c.at<float>(2, 0) = -8.0;
  c.at<float>(2, 1) = -32.0;
  c.at<float>(2, 2) = -18.0;

  for(size_t i=0; i<3; i++){
    for(size_t j=0; j<3; j++)
      std::cout<<c.at<float>(i,j)<<" ";
    std::cout<<std::endl;
  }
  
  cv::Mat y = cv::Mat::zeros(3, 1, CV_32FC1);
  int mode = 0;
  double lambda;

  lambda = powermethod(c, mode, y);
  std::cout<<"lambda = "<<lambda<<std::endl;

}
