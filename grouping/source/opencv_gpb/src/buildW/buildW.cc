#include "buildW.hh"

using namespace std;

namespace cv
{
  void buildW(const cv::Mat & input, 
	      cv::Mat & ind_x,
	      cv::Mat & ind_y,
	      cv::Mat & val)
  {
    int dthresh = 5;
    float sigma = 0.1;

    // copy edge info into lattice struct
    Group::DualLattice boundaries; 
    //int h = input.rows;
    //int w = input.cols;
    
    cv::copyMakeBorder(input, boundaries.H, 1, 0, 0, 0, cv::BORDER_CONSTANT, 0.0);
    cv::copyMakeBorder(input, boundaries.V, 0, 0, 1, 0, cv::BORDER_CONSTANT, 0.0);
    cv::transpose(boundaries.H, boundaries.H);
    cv::transpose(boundaries.V, boundaries.V);
    
    boundaries.width = boundaries.H.rows;
    boundaries.height = boundaries.V.cols;

    Group::SupportMap ic;
    Group::computeSupport(boundaries,dthresh,1.0f,ic);

    SMatrix* W = NULL;
    Group::computeAffinities2(ic,sigma,dthresh,&W);
    int nnz = 0;
    for(size_t i=0; i<W->n; i++)
      nnz += W->nz[i];
    ind_x = cv::Mat::ones(nnz, 1, CV_32SC1);
    ind_y = cv::Mat::ones(nnz, 1, CV_32SC1);
    val   = cv::Mat::zeros(nnz, 1, CV_32FC1);
    int ct = 0;
    for(size_t row = 0; row < W->n; row++){
      for(size_t i=0; i<W->nz[row]; i++){
	ind_y.at<int>(ct+i) = static_cast<int>(row);
	ind_x.at<int>(ct+i) = static_cast<int>(W->col[row][i]);
	  val.at<float>(ct+i) = static_cast<double>(W->values[row][i]);
      }
      ct = ct + W->nz[row];
    }
    delete W;
  }
  
}
