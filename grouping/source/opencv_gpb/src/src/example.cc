
#include "globalPb.hh"
#include "cv_lib_filters.hh"

using namespace std;
using namespace cv;
using namespace libFilters;

int main(int argc, char** argv){

  Mat img0, gPb, gPb_thin;// texton, mPb_max;
  vector<Mat> gPb_ori; 
  
  img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  globalPb(img0, gPb, gPb_thin, gPb_ori);
  
  imshow("Original", img0);
  imshow("gPb",  gPb);
  imshow("gPb_thin", gPb_thin);
  Display_EXP(gPb_ori, "gPb_ori", 4);
  while(true){
    char ch = waitKey(0);
    if(ch == 27) break;
  }
}
