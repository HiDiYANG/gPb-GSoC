
#include "globalPb.hh"
#include "cv_lib_filters.hh"

using namespace std;
using namespace cv;
using namespace libFilters;

int main(int argc, char** argv){

  Mat kernel, img0, img1, texton;
  vector<Mat> layers, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20; 
  vector<Mat> cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20;
   
  img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  globalPb(img0, img1);
  MakeFilter(3, 1.5708, kernel);
  multiscalePb(img0, layers);
  pb_parts_final_selected(layers, texton, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20);
  
  imshow("Original", img0);
  imshow("texton", texton*4);
  Display_EXP(bg_r3, "bg_r3");
  Display_EXP(bg_r5, "bg_r5");
  Display_EXP(bg_r10, "bg_r10");
  Display_EXP(cga_r5, "cga_r5");
  Display_EXP(cga_r10, "cga_r10");
  Display_EXP(cga_r20, "cga_r20");
  Display_EXP(cgb_r5, "cgb_r5");
  Display_EXP(cgb_r10, "cgb_r10");
  Display_EXP(cgb_r20, "cgb_r20");
  Display_EXP(tg_r5, "tg_r5");
  Display_EXP(tg_r10, "tg_r10");
  Display_EXP(tg_r20, "tg_r20");
  while(true){
    char ch = waitKey(0);
    if(ch == 27) break;
  }
}
