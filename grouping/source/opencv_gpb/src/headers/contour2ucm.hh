#include <iostream>
#include <vector>
#include <stack>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include "watershed.h"

class contour_vertex;
class contour_edge;

class contour_vertex{
public:
  int id;
  bool is_subdivision;
  int x;
  int y;
  std::vector<contour_edge> edges_start;
  std::vector<contour_edge> edges_end;

  //------- vertex method ---------

  cv::Point point() const;
};

class contour_edge{
public:
  int id;
  int contour_equiv_id;
  bool is_completion;
  std::vector<int> x_coords;
  std::vector<int> y_coords;
  contour_vertex * vertex_start;
  contour_vertex * vertex_end;
  int vertex_start_enum;
  int vertex_end_enum;

  //------- edge method -----------

  int size() const;
  double length() const;
  cv::Point point(int) const;
};



namespace cv
{
  void creat_finest_partition(const cv::Mat & gPb,
			      cv::Mat & ws_wt,
			      cv::Mat & labels,
			      cv::Mat & _is_vertex,
			      cv::Mat & _is_edge);

  void contour2ucm(const cv::Mat & gPb,
		   const vector<cv::Mat> & gPb_ori,
		   cv::Mat & labels,
		   cv::Mat & _is_vertex,
		   cv::Mat & _is_edge);
 
}
