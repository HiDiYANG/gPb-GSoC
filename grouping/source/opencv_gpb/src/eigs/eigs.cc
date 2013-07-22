#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

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
	     cv::Mat & z,
	     const double tol)
{
    double norm, lambda, lambda_lst, err;
    int n = q.rows;
    z = cv::Mat::zeros(n, 1, CV_32FC1);
    bool fst_loop = true;
    
    do{
      z=a*q;
      norm = 0.0;
      for(size_t j=0; j<n; j++){
	norm += z.at<float>(j, 0)*z.at<float>(j, 0);
      }
      norm = sqrt(norm);
      for(size_t j=0; j<n; j++){
	q.at<float>(j, 0)=z.at<float>(j, 0)/norm;
      }

      if(fst_loop){
	lambda_lst = 0.0;
	fst_loop = false;
      }else
	lambda_lst = lambda;

      lambda = 0.0;
      for(size_t j=0; j<n; j++)
	lambda += q.at<float>(j, 0)*z.at<float>(j, 0);
      err = fabs(lambda - lambda_lst);
    }while(err > tol);
    return lambda;
}

double powermethod(const cv::Mat & x, 
		   int mode, 
		   cv::Mat & y,
		   double tol)
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
  lambda = power(a,q,z,tol);
    
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
    
    for(size_t i=0; i<4; i++){
      for(size_t j=0; j<4; j++)
	std::cout<<b.at<float>(i,j)<<" ";
      std::cout<<std::endl;
    }
    
    
    /* perform one more time power iteration method */
    q = cv::Mat::zeros(x.rows,1,CV_32FC1);
    q.at<float>(0,0)=1.0;
            
    /* iterative method to find maximum eigenvalue */
    lambdamin = power(b,q,z,tol);
    
    cv::multiply(q, ones, y, -1.0);
    w = lambda - lambdamin;
  } 
  return w;
}

int main(int argc, char** agrv)
{
  cv::Mat c = cv::Mat::zeros(4, 4, CV_32FC1);
  c.at<float>(0, 0) = 4.0;
  c.at<float>(0, 1) = 2.0;
  c.at<float>(0, 2) = 2.0;
  c.at<float>(0, 3) = 1.0;

  c.at<float>(1, 0) = 2.0;
  c.at<float>(1, 1) = -3.0;
  c.at<float>(1, 2) = 1.0;
  c.at<float>(1, 3) = 5.0;
  
  c.at<float>(2, 0) = 2.0;
  c.at<float>(2, 1) = 1.0;
  c.at<float>(2, 2) = 3.0;
  c.at<float>(2, 3) = 1.0;

  c.at<float>(3, 0) = 1.0;
  c.at<float>(3, 1) = 5.0;
  c.at<float>(3, 2) = 1.0;
  c.at<float>(3, 3) = 2.0;

  for(size_t i=0; i<4; i++){
    for(size_t j=0; j<4; j++)
      std::cout<<c.at<float>(i,j)<<" ";
    std::cout<<std::endl;
  }
  
  cv::Mat y = cv::Mat::zeros(4, 1, CV_32FC1);
  int mode = 1;
  double lambda;
  double tol = 0.00001;

  lambda = powermethod(c, mode, y, tol);
  std::cout<<"lambda = "<<lambda<<std::endl;

}
