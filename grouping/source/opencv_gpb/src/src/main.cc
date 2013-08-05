
#include "globalPb.hh"
#include "cv_lib_filters.hh"
#include "contour2ucm.hh"

using namespace std;
using namespace libFilters;

int main(int argc, char** argv){

  cv::Mat img0, gPb, gPb_thin, ucm;
  vector<cv::Mat> gPb_ori; 

  img0 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::globalPb(img0, gPb, gPb_thin, gPb_ori);
  cv::contour2ucm(gPb, gPb_ori, ucm);

  cv::imshow("Original", img0);
  cv::imshow("gPb",  gPb);
  cv::imshow("gPb_thin", gPb_thin);
  cv::imshow("ucm", ucm);
  while(true){
    char ch = cv::waitKey(0);
    if(ch == 27) break;

  }
  
}
