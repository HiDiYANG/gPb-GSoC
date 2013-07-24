//
//    contour2ucm
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 22/07/13.
//    Copyright (c) 2013 The Australian National University. 
//    and Willow Garage inc.
//    All rights reserved.
//    
//

#include "contour2ucm.hh"
#define MAX_SIZE 100

using namespace std;

namespace cv
{
  int connected_component(const cv::Mat & ws_bw,
			   cv::Mat & labels)
  {
    cv::Mat mask = cv::Mat::ones(3, 3, CV_32SC1);
    labels = cv::Mat::zeros(ws_bw.rows, ws_bw.cols, CV_32SC1);
    int dims_bw[2]={ws_bw.rows, ws_bw.cols};
    int dims_msk[2]={mask.rows, mask.cols};
    int next_label = 1;
    int cont = 1;

    stack<int*> q;
    int* pos = new int[2];
    int* curr_pos = new int[2];

    for(size_t i=0; i<ws_bw.rows; i++)
      for(size_t j=0; j<ws_bw.cols; j++){
	int val = ws_bw.at<int>(i,j);
	if((val != 0) && 
	   (labels.at<int>(i,j) == 0)){	   
	  labels.at<int>(i,j)=next_label;
	  
	  //record current position
	  pos[0] = i; pos[1] = j;
	  q.push(pos);
	  while(q.size()){
	    curr_pos = q.top(); //load current position
	    q.pop();            //clear current position from stack
	    
	    //initial mask matrix
	    for(size_t m_i=0; m_i<mask.rows; m_i++)
	      for(size_t m_j=0; m_j<mask.cols; m_j++)
		mask.at<int>(m_i, m_j)=1;
	    mask.at<int>(1,1)=0;
	    
	    //reset mask matrix according to current position ...
	    if(curr_pos[1]>=ws_bw.cols-1){
	      int m_j = mask.cols-1;
	      for(size_t m_i=0; m_i<mask.rows; m_i++)
		mask.at<int>(m_i, m_j) = 0;
	    }
	    if(curr_pos[0]>=ws_bw.rows-1){
	      int m_i = mask.rows-1;
	      for(size_t m_j=0; m_j<mask.cols; m_j++)
		mask.at<int>(m_i, m_j) = 0;
	    }

	    if(curr_pos[1]<= 0){
	      int m_j = 0;
	      for(size_t m_i=0; m_i<mask.rows; m_i++)
		mask.at<int>(m_i, m_j) = 0;
	    }
	    if(curr_pos[0] <= 0){
	      int m_i = 0;
	      for(size_t m_j=0; m_j<mask.cols; m_j++)
		mask.at<int>(m_i, m_j) = 0;
	    }

	    //check neighborhood
	    for(size_t m_i=0; m_i<mask.rows; m_i++)
	      for(size_t m_j=0; m_j<mask.cols; m_j++){
		if(mask.at<int>(m_i, m_j) == 1){
		  int ind_x = curr_pos[0]+m_i-1;
		  int ind_y = curr_pos[1]+m_j-1;
		  int* neigh_pos = new int[2];
		  if((ws_bw.at<int>(ind_x, ind_y) == val) &&
		     (labels.at<int>(ind_x, ind_y) == 0)){
		    labels.at<int>(ind_x, ind_y) = next_label;
		    neigh_pos[0] = ind_x; neigh_pos[1] = ind_y;
		    q.push(neigh_pos);
		  }
		}
	      }
	  }//end_while
	  next_label++;
	}
      }
    
    return next_label;
  }

  void contour_side(const cv::Mat & ws_bw,
		    cv::Mat & labels)
  {
  
  }
  
  void creat_finest_partition(const cv::Mat & gPb,
			      cv::Mat & ws_wt)
  {
    cv::Mat temp = cv::Mat::zeros(gPb.rows, gPb.cols, CV_32FC1);
    cv::Mat ones = cv::Mat::ones(gPb.rows, gPb.cols, CV_32FC1);
    cv::multiply(gPb, ones, temp, 255.0);
    temp.convertTo(temp, CV_64FC1);
    cv::watershedFull(temp, 1, ws_wt);
  }

  void contour2ucm(const cv::Mat & gPb,
		   const vector<cv::Mat> & gPb_ori,
		   cv::Mat & labels)
  { 
    cv::Mat ws_wt;
    creat_finest_partition(gPb, ws_wt);
    ws_wt.convertTo(ws_wt, CV_8UC1);
    cv::Mat ws_bw = cv::Mat::zeros(ws_wt.rows, ws_wt.cols, CV_32SC1);
    
    for(size_t i=0; i<ws_wt.rows; i++)
      for(size_t j=0; j<ws_wt.cols; j++)
	if(ws_wt.at<char>(i,j)==0)
	  ws_bw.at<int>(i,j)=1;
    
    int num_labels;
    num_labels = connected_component(ws_bw, labels);
    cout<<"num_labels: "<<num_labels<<endl;
    for(size_t i=0; i<labels.rows; i++)
      for(size_t j=0; j<labels.cols; j++)
	labels.at<int>(i,j) *= 51; 
    labels.convertTo(labels, CV_8UC1);
    ws_bw.convertTo(ws_bw, CV_8UC1);
    imshow("ws_bw", ws_bw*255);
  }
}
