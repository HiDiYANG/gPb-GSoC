
#include "globalPb.hh"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

  Mat kernel, img0, img1, texton;
  Mat chR, chG, chB;
  vector<Mat> layers, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20;
   
  img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  globalPb(img0, img1);
  MakeFilter(3, 1.5708, kernel);
  multiscalePb(img0, layers);
  imshow("Original", img0);
  pb_parts_final_selected(layers, texton, bg_r3, bg_r5, bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20);

  imshow("texton", texton*4);
  while(true){
    char ch = waitKey(0);
    if(ch == 27) break;
  }
}
