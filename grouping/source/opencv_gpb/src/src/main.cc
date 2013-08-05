
#include "globalPb.hh"
#include "cv_lib_filters.hh"
#include "contour2ucm.hh"

using namespace std;
using namespace libFilters;

cv::Mat markers, ucm;
cv::Point prev_pt(-1, -1);
void on_mouse( int event, int x, int y, int flags, void* param )
{
  if( !ucm.empty() )
    return;
	
  if( event == CV_EVENT_LBUTTONUP || !(flags&CV_EVENT_FLAG_LBUTTON) )
    prev_pt = cv::Point(-1,-1);
  else if( event == CV_EVENT_LBUTTONDOWN )
    prev_pt = cv::Point(x,y);
  else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
  {
    CvPoint pt = cv::Point(x,y);
    if( prev_pt.x < 0 )
      prev_pt = pt;
    cv::line( markers, prev_pt, pt, cv::Scalar(255), 3, 8, 0 );
    cv::line( ucm,     prev_pt, pt, cv::Scalar(255), 3, 8, 0 );
    prev_pt = pt;
    imshow("ucm", ucm );
  }
}

int main(int argc, char** argv){

  cv::Mat img0, gPb, gPb_thin;
  vector<cv::Mat> gPb_ori; 

  img0 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::globalPb(img0, gPb, gPb_thin, gPb_ori);
  cv::contour2ucm(gPb, gPb_ori, ucm, DOUBLE_SIZE);

  markers = cv::Mat::zeros(ucm.rows, ucm.cols, CV_8UC1);
  cv::imshow("Original", img0);
  cv::imshow("gPb",  gPb);
  cv::imshow("gPb_thin", gPb_thin);
  cv::imshow("ucm", ucm);

  cv::setMouseCallback("ucm", on_mouse, 0);


  while(true){
    char ch = cv::waitKey(0);
    if(ch == 27) break;
  }
  
}
