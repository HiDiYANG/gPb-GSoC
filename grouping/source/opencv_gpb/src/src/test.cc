//#include "globalPb.hh"
//#include "cv_lib_filters.hh"
//#include "contour2ucm.hh"
#include <iostream>
#include <list>
#include <stack>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

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


static contour_vertex* create_contour_vertex(int x, int y)
{
  contour_vertex* v = new contour_vertex[1];
  v->id = 0;
  v->is_subdivision = false;
  v->x = x;
  v->y = y;
  return v;
}

static contour_edge* create_contour_edge(contour_vertex & v_start, 
					 contour_vertex & v_end)
{
  contour_edge* e = new contour_edge[1];
  e->id = 0;
  e->contour_equiv_id = 0;
  e->is_completion = false;
  e->vertex_start = &v_start;
  e->vertex_end   = &v_end;
  e->vertex_start_enum = v_start.edges_start.size();
  e->vertex_end_enum = v_end.edges_end.size();
  v_start.edges_start.push_back(*e);
  v_end.edges_end.push_back(*e);
  return e;
}

cv::Point contour_vertex::point() const{
  return cv::Point(this->x, this->y);
} 

int contour_edge::size() const{
  return this->x_coords.size();
}

cv::Point contour_edge::point(int n) const{
  return cv::Point(this->x_coords[n], this->y_coords[n]);
}

double contour_edge::length() const{
  cv::Point start = this->vertex_start->point();
  cv::Point end = this->vertex_end->point();
  double length = sqrt((end.x-start.x)*(end.x-start.x) + (end.y-start.y)*(end.y-start.y));
  return length;
}


using namespace std;

int main()
{
  vector<contour_vertex> _vertices;
  vector<contour_edge> _edges;
  
  contour_vertex * v1 = create_contour_vertex(1,1);
  contour_vertex * v2 = create_contour_vertex(3,3);

  _vertices.push_back(*v1);
  _vertices.push_back(*v2);

  contour_vertex * vv1 = &_vertices[0];
  contour_vertex * vv2 = &_vertices[1];

  contour_edge * e1 = create_contour_edge(*vv1, *vv2);
  cout<<"vv1.edges_start.empty: "<<vv1->edges_start.empty()<<endl;
  cout<<"vv2.edges_end.empty: "<<vv2->edges_end.empty()<<endl;
  cout<<"-------------------------------"<<endl;
  cout<<"v1.edges_start.empty: "<<_vertices[0].edges_start.empty()<<endl;
  cout<<"v2.edges_end.empty: "<<_vertices[1].edges_end.empty()<<endl;
  cout<<"-------------------------------"<<endl;
  
}
