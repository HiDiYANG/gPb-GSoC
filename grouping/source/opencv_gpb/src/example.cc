
#include "globalPb.hh"
using namespace std;
using namespace cv;

int main(int argc, char** argv){

   Mat kernel, img0, img1;
   Mat chR, chG, chB;
   vector<Mat> layers;
   
   img0 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   globalPb(img0, img1);
   MakeFilter(3, 1.5708, kernel);
   /*for(size_t i=0; i < kernel.rows; i++){
     for(size_t j=0; j < kernel.cols; j++)
       cout<<kernel.at<Vec3d>(i,j)[2]<<" ";
     cout<<endl;
     }*/
   multiscalePb(img0, layers);
   imshow("Original", img0);
   pb_parts_final_selected(layers);
   while(true){
     char ch = waitKey(0);
     if(ch == 27) break;
   }
}
