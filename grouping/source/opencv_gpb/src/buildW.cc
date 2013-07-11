#include "buildW.hh"

using namespace std;

namespace cv
{
  void buildW(const cv::Mat & input, 
	      cv::Mat & output)
  {
    int dthresh = 5;
    float sigma = 0.1;

    // copy edge info into lattice struct
    Group::DualLattice boundaries; 
    int h = input.rows;
    int w = input.cols;
    cout<<"h = "<<h<<endl;
    
    //cv::copyMakeBorder(input, boundaries.H, 1, 0, 0, 0, cv::BORDER_CONSTANT, 0.0);
    //cv::copyMakeBorder(input, boundaries.V, 0, 0, 1, 0, cv::BORDER_CONSTANT, 0.0);
    boundaries.width = boundaries.H.rows;
    boundaries.height = boundaries.V.cols;

    Group::SupportMap ic;
    Group::computeSupport(boundaries,dthresh,1.0f,ic);

    SMatrix* W = NULL;
    Group::computeAffinities2(ic,sigma,dthresh,&W);
  }
  
}
