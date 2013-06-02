
#include "globalPb.hh"
using namespace std;
using namespace cv;

int main(int argc, char** argv){

   Mat kernel, img0, img1;
   img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   globalPb(img0, img1);
   MakeFilter(3, 1.5708, kernel);
   for(size_t i=0; i < kernel.rows; i++){
     for(size_t j=0; j < kernel.cols; j++)
       cout<<kernel.at<Vec3d>(i,j)[2]<<" ";
     cout<<endl;
   }
}
