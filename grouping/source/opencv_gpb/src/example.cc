
#include "globalPb.hh"
#include "cv_lib_filters.hh"

using namespace std;
using namespace cv;
using namespace libFilters;

int main(int argc, char** argv){

  Mat img0, img1, texton, mPb_max;
  vector<Mat> layers, kernel, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20; 
  vector<Mat> cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20, mPb_all;
  
  img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  //globalPb(img0, img1);
  multiscalePb(img0, mPb_max, mPb_all, bg_r3,bg_r5, bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20);
  
  imshow("Original", img0);
  //Display_EXP(mPb_all, "mPb_all", 4);
  Display_EXP(mPb_max, "mPb_max");
  /*Display_EXP(bg_r5, "bg_r5", 4);
  Display_EXP(bg_r10, "bg_r10", 4);
  Display_EXP(cga_r5, "cga_r5", 4);
  Display_EXP(cga_r10, "cga_r10", 4);
  Display_EXP(cga_r20, "cga_r20", 4);
  Display_EXP(cgb_r5, "cgb_r5", 4);
  Display_EXP(cgb_r10, "cgb_r10", 4);
  Display_EXP(cgb_r20, "cgb_r20", 4);
  Display_EXP(tg_r5, "tg_r5", 4);
  Display_EXP(tg_r10, "tg_r10", 4);
  Display_EXP(tg_r20, "tg_r20", 4);*/
  while(true){
    char ch = waitKey(0);
    if(ch == 27) break;
  }
}
