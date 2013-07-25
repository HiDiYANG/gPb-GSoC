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
typedef struct{
  int id;
  bool is_subdivision;
  int x;
  int y;
}vertex;

namespace cv
{
  void neighbor_exists_2D(const int* pos,
			  const int size_x, const int size_y,
			  cv::Mat & mask)
  {
    //initial mask matrix
    mask = cv::Mat::ones(3, 3, CV_32SC1);
    mask.at<int>(1,1) = 0;
	    
    //reset mask matrix according to current position ...
    if(pos[1]>=size_y-1){
      int j = mask.cols-1;
      for(size_t i=0; i<mask.rows; i++)
	mask.at<int>(i, j) = 0;
    }
    if(pos[0]>=size_x-1){
      int i = mask.rows-1;
      for(size_t j=0; j<mask.cols; j++)
	mask.at<int>(i, j) = 0;
    }

    if(pos[1]<= 0){
      int j = 0;
      for(size_t i=0; i<mask.rows; i++)
	mask.at<int>(i, j) = 0;
    }
    if(pos[0] <= 0){
      int i = 0;
      for(size_t j=0; j<mask.cols; j++)
	mask.at<int>(i, j) = 0;
    }
  }

  void neighbor_compare_2D(const int* pos,
			   const cv::Mat & labels,
			   const cv::Mat & mask,
			   cv::Mat & mask_cmp)
  {
    mask_cmp = cv::Mat::zeros(3, 3, CV_32SC1);
    for(size_t i=0; i<mask.rows; i++)
      for(size_t j=0; j<mask.cols; j++){
	int val = labels.at<int>(pos[0], pos[1]);
	if(mask.at<int>(i, j)==1)
	  if(labels.at<int>(pos[0]+i-1, pos[1]+j-1)==val)
	    mask_cmp.at<int>(i,j)=1;
      }	  
  }

  bool is_vertex_2D(const cv::Mat & mask_cmp)
  {
    int n, s, e, w, ne, nw, se, sw;
    n = mask_cmp.at<int>(1,2);
    s = mask_cmp.at<int>(1,0);
    e = mask_cmp.at<int>(2,1);
    w = mask_cmp.at<int>(0,1);

    ne = mask_cmp.at<int>(2,2);
    nw = mask_cmp.at<int>(0,2);
    se = mask_cmp.at<int>(2,0);
    sw = mask_cmp.at<int>(0,0);

    int n_adjacent = n + s + e + w;
    int n_diagonal = ne+ nw+ se+ sw;
    int nn = n_adjacent + n_diagonal;

    if(nn<=1)
      return true;
    else if(nn==2){
      if((n && (ne || nw)) ||
	 (s && (se || sw)) ||
	 (e && (ne || se)) ||
	 (w && (nw || sw)))
	return true;
      else
	return false;
    }
    else if((nn == 3) || (nn == 4)){
      if((nw && n && w) || (n && ne && e) ||
	 (w && sw && s) || (e && s && se))
	return true;
      else if
	((nw && n && ne)|| (sw && s && se)||
	 (nw && w && sw)|| (ne && e && se))
	return (nn==3);
      else if(n_adjacent >= 3 )
	return true;
      else if(n_diagonal >= 3 )
	return true;
      else if
	((n && se && sw)|| (s && ne && nw)||
	 (w && se && ne)|| (e && nw && sw))
	return true;
      else if
	((nw && s && e) || (ne && s && w) ||
	 (sw && n && e) || (se && n && w))
	return true;
      else
	return false;
    }
    else
      return(!((n_adjacent == 2) && ((n && s) || (e && w))));

  }

  int connected_component(const cv::Mat & ws_bw,
			   cv::Mat & labels)
  {
    cv::Mat mask;
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
	    neighbor_exists_2D(curr_pos, ws_bw.rows, ws_bw.cols, mask);
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
    next_label--;
    return next_label;
  }

  void contour_side(const cv::Mat & ws_bw,
		    cv::Mat & labels,
		    cv::Mat & is_v,
		    cv::Mat & assign)
  {
    int num_labels = connected_component(ws_bw, labels);
    int rows = ws_bw.rows, cols = ws_bw.cols;
    int* label_x = new int[num_labels];
    int* label_y = new int[num_labels];
    int* pos = new int[2];

    vertex * v = new vertex[1];
    cv::Mat mask, mask_cmp;
    stack<vertex *> _vertices;
    
    is_v = cv::Mat::zeros(rows, cols, CV_32SC1);
    //ws_bw.copyTo(is_e);
    assign = cv::Mat::zeros(rows, cols, CV_32SC1);

    for(size_t i=0; i<rows; i++)
      for(size_t j=0; j<cols; j++){
	int val = labels.at<int>(i,j);
	if(val != 0){
	  pos[0] = i; pos[1] = j;
	  neighbor_exists_2D(pos, rows, cols, mask);
	  neighbor_compare_2D(pos, labels, mask, mask_cmp);
	  if(is_vertex_2D(mask_cmp)){
	    is_v.at<int>(i,j) = 255;
	    cout<<"is_v: "<<is_v.at<int>(i,j)<<endl;
	    v->id = _vertices.size();
	    v->x = i; v->y = j; v->is_subdivision = false;
	    assign.at<int>(i,j) = v->id;
	    _vertices.push(v); 
	    cout<<"v.id = "<<v->id<<endl;
	  }
	  label_x[val-1]=i;
	  label_y[val-1]=j;
	}
      }

  }
  
  void fit_contour(const cv::Mat & ws_bw,
		   cv::Mat & labels,
		   cv::Mat & is_v,
		   cv::Mat & assign)
  {
    contour_side(ws_bw, labels, is_v, assign);
  }

  void creat_finest_partition(const cv::Mat & gPb,
			      cv::Mat & ws_wt,
			      cv::Mat & labels,
			      cv::Mat & is_v,
			      cv::Mat & assign)
  {
    cv::Mat temp = cv::Mat::zeros(gPb.rows, gPb.cols, CV_32FC1);
    cv::Mat ws_bw = cv::Mat::ones(gPb.rows, gPb.cols, CV_32FC1);
    cv::multiply(gPb, ws_bw, temp, 255.0);
    temp.convertTo(temp, CV_64FC1);
    cv::watershedFull(temp, 1, ws_wt);
    
    for(size_t i=0; i<ws_wt.rows; i++)
      for(size_t j=0; j<ws_wt.cols; j++)
	if(ws_wt.at<int>(i,j) > 0)
	  ws_bw.at<int>(i,j)=0;
    
    fit_contour(ws_bw, labels, is_v, assign);
  }

  void contour2ucm(const cv::Mat & gPb,
		   const vector<cv::Mat> & gPb_ori,
		   cv::Mat & labels)
  { 
    cv::Mat ws_wt, is_v, assign;
    creat_finest_partition(gPb, ws_wt, labels, is_v, assign);
    //ws_wt.convertTo(ws_wt, CV_8UC1);
    cv::Mat ws_bw = cv::Mat::zeros(ws_wt.rows, ws_wt.cols, CV_32SC1);
    fit_contour(ws_bw, labels, is_v, assign);
    for(size_t i=0; i<ws_wt.rows; i++)
      for(size_t j=0; j<ws_wt.cols; j++)
	if(ws_wt.at<int>(i,j)==0)
	  ws_bw.at<int>(i,j)=1;

    int num_labels = connected_component(ws_bw, labels);
    
    int scale = 255/(num_labels);
    cout<<"num_labels: "<<num_labels<<endl;
    for(size_t i=0; i<labels.rows; i++)
      for(size_t j=0; j<labels.cols; j++)
	labels.at<int>(i,j) *= scale; 
    labels.convertTo(labels, CV_8UC1);    
    is_v.convertTo(is_v, CV_8UC1);
    imshow("is_v", is_v);
	
  }
}
