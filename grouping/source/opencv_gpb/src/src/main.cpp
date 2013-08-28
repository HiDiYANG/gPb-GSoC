//
//    gPb opencv implementation
//    including interactive segmentation.
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University. 
//    and Willow Garage inc.
//    All rights reserved.
//    
//

#include "globalPb.h"
#include "contour2ucm.h"

using namespace std;

cv::Mat markers, ucm2;
cv::Point prev_pt(-1, -1);
void on_mouse( int event, int x, int y, int flags, void* param )
{
  if( ucm2.empty() )
    return;
	
  if( event == cv::EVENT_LBUTTONUP || !(flags& cv::EVENT_FLAG_LBUTTON) )
    prev_pt = cv::Point(-1,-1);
  else if( event == cv::EVENT_LBUTTONDOWN )
    prev_pt = cv::Point(x,y);
  else if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) )
  {
    cv::Point pt = cv::Point(x,y);
    if( prev_pt.x < 0 )
      prev_pt = pt;
    cv::line( markers, prev_pt, pt, uchar(255), 3, 8, 0 );
    cv::line( ucm2,     prev_pt, pt, 1.0, 3, 8, 0 );
    prev_pt = pt;
    cv::imshow("ucm", ucm2 );
  }
}

int main(int argc, char** argv){

  //info block
  system("clear");
  cout<<"(before running it, roughly mark the areas on the ucm window)"<<endl;
  cout<<"Press 'r' - resort the original ucm, and remark"<<endl;
  cout<<"Press 'w' or 'ENTER' - conduct interactive segmentation"<<endl;
  cout<<"Press 'ESC' - exit the program"<<endl<<endl<<endl;

  cv::Mat img0, gPb, gPb_thin, ucm;
  vector<cv::Mat> gPb_ori; 

  img0 = cv::imread(argv[1], -1);
  cv::globalPb(img0, gPb, gPb_thin, gPb_ori);

  // if you wanna conduct interactive segmentation later, choose DOUBLE_SIZE, otherwise SINGLE_SIZE will do either.
  cv::contour2ucm(gPb, gPb_ori, ucm, DOUBLE_SIZE);
  
  //back up
  ucm.copyTo(ucm2);
  markers = cv::Mat::zeros(ucm.size(), CV_8UC1);
  
  cv::imshow("Original", img0);
  cv::imshow("gPb",  gPb);
  cv::imshow("gPb_thin", gPb_thin);
  cv::imshow("ucm", ucm2);
  cv::setMouseCallback("ucm", on_mouse, 0);

  while(true){
    char ch = cv::waitKey(0);
    if(ch == 27) break;
    
    if(ch == 'r'){
      //restore everything
      markers = cv::Mat::zeros(markers.size(), CV_8UC1);
      ucm.copyTo(ucm2);
      cv::imshow("ucm", ucm2);
      cv::destroyWindow("boundary");
      cv::destroyWindow("labels");
    }

    if(ch == 'w' || ch == '\n'){
      cv::Mat boundary, labels, seeds;
      vector< vector<cv::Point> > contours;
      vector<cv::Vec4i> hierarchy;
      cv::findContours(markers, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
      seeds = cv::Mat::zeros(markers.size(), CV_8UC1);
      int num_seed = 1;
      for( int i = 0; i< contours.size(); i++ ){ 
	cv::drawContours(seeds, contours, i, uchar(num_seed++), -1, 8, hierarchy, 0, cv::Point() );
      }
      seeds.convertTo(seeds, CV_32SC1);
      cv::uvt(ucm, seeds, boundary, labels, SINGLE_SIZE);
      cv::imshow("boundary", boundary*255);
      cv::imshow("labels", labels*int(255/num_seed));
    }
      
  }
  //clean up
  img0.release();
  gPb.release();
  gPb_thin.release();
  ucm.release();
  ucm2.release();
  markers.release();
  gPb_ori.clear();
}
